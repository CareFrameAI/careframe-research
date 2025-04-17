from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                              QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                              QSplitter, QTabWidget, QTextEdit, QComboBox,
                              QFormLayout, QTreeWidget, QTreeWidgetItem, QScrollArea,
                              QSizePolicy, QDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QItemSelectionModel, QByteArray
from PyQt6.QtGui import QFont, QBrush, QColor
from PyQt6.QtSvgWidgets import QSvgWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the icon loading function
from helpers.load_icon import load_bootstrap_icon

import pandas as pd
import json
import base64
from datetime import datetime
from llms.client import (call_claude_sync, call_claude_async, 
                         call_claude_with_image_sync, call_claude_with_image_async,
                         call_claude_with_multiple_images_async,
                         call_llm_sync, call_llm_async, call_llm_vision_sync, call_llm_vision_async)
import io
import asyncio
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                              QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                              QSplitter, QTabWidget, QTextEdit, QComboBox,
                              QFormLayout, QTreeWidget, QTreeWidgetItem, QScrollArea,
                              QSizePolicy, QDialog, QMenu, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QItemSelectionModel, QByteArray, QThread
from PyQt6.QtGui import QFont, QBrush, QColor, QCursor


class TestEvaluationWidget(QWidget):
    """Widget for evaluating test results."""
    
    # Signal to notify when active study changes
    active_study_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.studies_manager = None
        self.is_dark_mode = False  # Default to light mode
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Top section container - as compact as possible
        top_section = QWidget()
        top_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        
        # Header and study selection in one row
        header_label = QLabel("Statistical Test Evaluation")
        header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        header_label.setFont(font)
        # Add icon to header
        header_label.setPixmap(load_bootstrap_icon("graph-up", size=24).pixmap(24, 24))
        top_layout.addWidget(header_label)
        
        # Study selection with label
        study_selection = QWidget()
        study_layout = QHBoxLayout(study_selection)
        study_layout.setContentsMargins(0, 0, 0, 0)
        study_layout.setSpacing(5)
        
        study_label = QLabel("Study:")
        study_layout.addWidget(study_label)
        
        self.studies_combo = QComboBox()
        self.studies_combo.setMinimumWidth(400)
        self.studies_combo.currentIndexChanged.connect(self.on_study_selected_combo)
        study_layout.addWidget(self.studies_combo)
        
        self.refresh_btn = QPushButton("")
        self.refresh_btn.setFixedWidth(70)
        self.refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        self.refresh_btn.clicked.connect(self.refresh_studies_list)
        study_layout.addWidget(self.refresh_btn)
        
        top_layout.addWidget(study_selection)
        top_layout.addStretch()
        
        main_layout.addWidget(top_section)
        
        # Outcome selection row - compact
        outcome_section = QWidget()
        outcome_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        outcome_layout = QHBoxLayout(outcome_section)
        outcome_layout.setContentsMargins(0, 0, 0, 0)
        outcome_layout.setSpacing(5)
        
        outcome_label = QLabel("Outcome:")
        outcome_label.setPixmap(load_bootstrap_icon("clipboard-data", size=16).pixmap(16, 16))
        outcome_layout.addWidget(outcome_label)
        
        self.outcomes_combo = QComboBox()
        self.outcomes_combo.setMinimumWidth(200)
        self.outcomes_combo.currentIndexChanged.connect(self.on_outcome_selected)
        outcome_layout.addWidget(self.outcomes_combo)
        outcome_layout.addStretch()
        
        main_layout.addWidget(outcome_section)
        outcome_section.setVisible(False)
        self.outcomes_section = outcome_section
        
        # Create horizontal splitter for summary tree and visualizations
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Left side - Summary Tree
        tree_container = QWidget()
        tree_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(2)
        
        # Add tree header with icon
        tree_header = QLabel("Test Results")
        tree_header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        tree_header.setPixmap(load_bootstrap_icon("list-check", size=16).pixmap(16, 16))
        tree_layout.addWidget(tree_header)
        
        self.summary_tree = QTreeWidget()
        self.summary_tree.setHeaderLabels(["Test Parameter", "Result"])
        self.summary_tree.setColumnWidth(0, 300)
        self.summary_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tree_layout.addWidget(self.summary_tree)
        
        content_splitter.addWidget(tree_container)
        
        # Right side - Visualizations
        viz_container = QWidget()
        viz_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(2)
        
        # Add visualization header with icon
        viz_header = QLabel("Visualizations")
        viz_header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        viz_header.setPixmap(load_bootstrap_icon("bar-chart", size=16).pixmap(16, 16))
        viz_layout.addWidget(viz_header)
                
        # Add context menu to the summary tree
        self.summary_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.summary_tree.customContextMenuRequested.connect(self.show_tree_context_menu)
        
        # Create a new row for analyze buttons with better spacing
        analyze_buttons_container = QWidget()
        analyze_buttons_layout = QHBoxLayout(analyze_buttons_container)
        analyze_buttons_layout.setContentsMargins(0, 5, 0, 10)
        analyze_buttons_layout.setSpacing(10)
        
        # Add button to analyze all tree items with LLM
        analyze_tree_btn = QPushButton("Analyze All Test Results")
        analyze_tree_btn.setIcon(load_bootstrap_icon("list-check"))
        analyze_tree_btn.clicked.connect(self.analyze_all_tree_items_with_llm)
        analyze_buttons_layout.addWidget(analyze_tree_btn)
        
        # Add button to analyze all figures with LLM
        analyze_figures_btn = QPushButton("Analyze All Visualizations")
        analyze_figures_btn.setIcon(load_bootstrap_icon("bar-chart"))
        analyze_figures_btn.clicked.connect(self.analyze_all_figures_with_llm)
        analyze_buttons_layout.addWidget(analyze_figures_btn)
        
        # Add to the layout
        viz_layout.addWidget(analyze_buttons_container)
        
        # Remove or replace the existing single analyze button
        # viz_layout.addWidget(analyze_btn)  # Remove this line
        
        self.viz_scroll = QScrollArea()
        self.viz_scroll.setWidgetResizable(True)
        self.viz_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.viz_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.viz_scroll.setWidget(self.viz_container)
        viz_layout.addWidget(self.viz_scroll)
        
        content_splitter.addWidget(viz_container)
        content_splitter.setSizes([500, 500])
        
        main_layout.addWidget(content_splitter, stretch=1)
        
    def set_studies_manager(self, studies_manager):
        """Set the studies manager and refresh the studies list."""
        self.studies_manager = studies_manager
        self.refresh_studies_list()
        
    def refresh_studies_list(self):
        """Refresh the list of studies."""
        if not self.studies_manager:
            return
            
        studies = self.studies_manager.list_studies()
        self.studies_combo.clear()
        
        for study in studies:
            display_text = f"{study['name']} {'(Active)' if study['is_active'] else ''}"
            self.studies_combo.addItem(load_bootstrap_icon("file-earmark-text"), display_text, study["id"])
        
    def on_study_selected_combo(self, index):
        """Handle selection of a study in the combo box."""
        if index < 0:
            return
        
        study_id = self.studies_combo.itemData(index)
        study = self.studies_manager.get_study(study_id)
        
        if study:
            self.display_study_outcomes(study)
    
    def display_study_outcomes(self, study):
        """Display outcomes available for the selected study."""
        # Update outcomes dropdown
        self.outcomes_combo.clear()
        
        # Initialize outcomes_with_tests list
        outcomes_with_tests = []
        
        if hasattr(study, 'results') and study.results:
            # Collect all outcomes with test results
            for result in study.results:
                if (hasattr(result, 'test_results') and 
                    result.test_results and 
                    len(result.test_results) > 0):
                    outcomes_with_tests.append(result)
            
            if outcomes_with_tests:
                for result in outcomes_with_tests:
                    self.outcomes_combo.addItem(load_bootstrap_icon("graph-up"), result.outcome_name, result)
                self.outcomes_section.setVisible(True)
            else:
                self.outcomes_combo.addItem(load_bootstrap_icon("exclamation-triangle", color="#FFA500"), "No test results available")
                self.outcomes_section.setVisible(False)
        else:
            self.outcomes_combo.addItem(load_bootstrap_icon("exclamation-triangle", color="#FFA500"), "No test results available")
            self.outcomes_section.setVisible(False)
    
    def on_outcome_selected(self, index):
        """Handle selection of an outcome in the combo box."""
        if index < 0 or not self.outcomes_combo.itemData(index):
            return
        
        result = self.outcomes_combo.itemData(index)
        self.display_test_results(result)
    
    def display_test_results(self, result):
        """Display test results for the selected outcome."""
        if not result or not hasattr(result, 'test_results') or not result.test_results:
            return
        
        # Clear existing content
        self.summary_tree.clear()
        
        # Clear visualization area
        self.clear_visualization_tab()
        
        # Parse and display the test results
        parsed_result = self.parse_test_result(result.test_results)
        
        # Add basic test information to parsed results
        parsed_result['outcome_name'] = result.outcome_name
        parsed_result['dataset_name'] = result.dataset_name
        
        # Build the summary tree with all results
        self.build_summary_tree(parsed_result)
        
        # Extract and display figures if available
        if 'figures' in parsed_result and parsed_result['figures']:
            self.add_visualizations(parsed_result['figures'])

    def _get_result_color(self, value):
        """Get color for result values."""
        if isinstance(value, bool):
            return QColor("green") if value else QColor("red")
        elif isinstance(value, str):
            value = value.lower()
            if value in ['passed', 'true', 'yes']:
                return QColor("green")
            elif value in ['failed', 'false', 'no']:
                return QColor("red")
            elif value == 'warning':
                return QColor("orange")
        return QColor("black")

    def _get_assumption_color(self, result):
        """Get color for assumption results."""
        result = str(result).lower()
        if result in ['passed', 'true']:
            return QColor("green")
        elif result in ['failed', 'false']:
            return QColor("red")
        elif result == 'warning':
            return QColor("orange")
        return QColor("gray")

    def _format_value(self, value):
        """Format values appropriately based on their type."""
        if value is None:
            return 'N/A'
        elif isinstance(value, bool):
            return 'Yes' if value else 'No'
        elif isinstance(value, (int, float)):
            if pd.isna(value) or value in [float('inf'), float('-inf')]:
                return 'N/A'
            elif isinstance(value, float):
                # Use scientific notation for extreme values
                if abs(value) < 0.0001 or abs(value) > 10000:
                    return f"{value:.2e}"
                else:
                    return f"{value:.4f}"
            return str(value)
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    def build_summary_tree(self, test_result):
        """Build a dynamic summary tree from test results."""
        self.summary_tree.clear()
        
        def add_dict_to_tree(parent_item, data, key_prefix=''):
            """Recursively add dictionary items to tree."""
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'figures':  # Skip figures; handled separately
                        continue
                    if key == 'assumptions':  # Skip assumptions; handled in dedicated section
                        continue
                        
                    display_key = str(key).replace('_', ' ').title()
                    
                    # Create tree item with appropriate icon based on context
                    if parent_item is None:
                        item = QTreeWidgetItem(self.summary_tree, [display_key, ''])
                        # Add icons for top-level categories
                        if key == 'coefficients_structured':
                            item.setIcon(0, load_bootstrap_icon("table"))
                        elif key == 'descriptive_stats':
                            item.setIcon(0, load_bootstrap_icon("clipboard-data"))
                    else:
                        item = QTreeWidgetItem(parent_item, [display_key, ''])
                    
                    # Handle different types of values
                    if isinstance(value, dict):
                        add_dict_to_tree(item, value)
                    elif isinstance(value, (list, pd.DataFrame)):
                        add_tabular_data(item, value, key)
                    else:
                        add_value_to_item(item, value, key)
        
        def add_tabular_data(parent_item, data, key):
            """Add tabular data to tree with improved formatting for lists."""
            if isinstance(data, pd.DataFrame):
                # Display DataFrame in a table-like structure
                parent_item.setIcon(0, load_bootstrap_icon("table"))
                for col in data.columns:
                    col_item = QTreeWidgetItem(parent_item, [str(col), ''])
                    for idx, value in enumerate(data[col]):
                        row_item = QTreeWidgetItem(col_item, [f"Row {idx}", self._format_value(value)])
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    # List of dictionaries (e.g., coefficients)
                    parent_item.setIcon(0, load_bootstrap_icon("list"))
                    for i, item_dict in enumerate(data):
                        list_item = QTreeWidgetItem(parent_item, [f"Item {i+1}", ''])
                        add_dict_to_tree(list_item, item_dict)
                else:
                    # For simple lists, add each element as a separate child with a bullet
                    if len(data) == 1:
                        parent_item.setText(1, self._format_value(data[0]))
                    else:
                        parent_item.setIcon(0, load_bootstrap_icon("list-ul"))
                        for idx, item in enumerate(data):
                            child_item = QTreeWidgetItem(parent_item, [f"Item {idx+1}", f"• {self._format_value(item)}"])
        
        def add_value_to_item(item, value, key):
            """Add formatted value to tree item with appropriate styling."""
            formatted_value = self._format_value(value)
            item.setText(1, formatted_value)
            
            if key in ['significant', 'passed', 'failed']:
                color = self._get_result_color(value)
                item.setForeground(1, QBrush(color))
                
                # Add icon based on boolean/result value
                if value in [True, 'passed', 'PASSED', 'yes', 'YES', 'true', 'TRUE']:
                    item.setIcon(0, load_bootstrap_icon("check-circle", color="#28a745"))
                elif value in [False, 'failed', 'FAILED', 'no', 'NO', 'false', 'FALSE']:
                    item.setIcon(0, load_bootstrap_icon("x-circle", color="#dc3545"))
            elif key == 'p_value':
                alpha = test_result.get('alpha', 0.05)
                try:
                    p_val = float(value)
                    color = QColor("green") if p_val < alpha else QColor("red")
                    item.setForeground(1, QBrush(color))
                    
                    # Add significance icon
                    if p_val < alpha:
                        item.setIcon(0, load_bootstrap_icon("check-circle", color="#28a745"))
                    else:
                        item.setIcon(0, load_bootstrap_icon("x-circle", color="#dc3545"))
                except (ValueError, TypeError):
                    pass
            elif key == 'warnings':
                item.setIcon(0, load_bootstrap_icon("exclamation-triangle", color="#FFA500"))
                
        # Build the tree starting with the test_result dictionary
        add_dict_to_tree(None, test_result)
        
        self.summary_tree.expandAll()
        self.summary_tree.resizeColumnToContents(0)
        self.summary_tree.resizeColumnToContents(1)

    def add_visualizations(self, figures):
        """Add visualization figures to the visualization area."""
        if not figures:
            label = QLabel("No visualizations available for this test")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: gray; font-style: italic; margin: 20px;")
            label.setPixmap(load_bootstrap_icon("image", size=32).pixmap(32, 32))
            self.viz_layout.addWidget(label)
            return
        
        added_figures = 0
        
        for fig_name, fig in figures.items():
            if fig is not None:
                try:
                    title = fig_name.replace('_', ' ').title()
                    fig_widget = self.create_figure_widget(fig, title)
                    if fig_widget:
                        self.viz_layout.addWidget(fig_widget)
                        added_figures += 1
                except Exception as e:
                    print(f"Error adding visualization '{fig_name}': {e}")
        
        if added_figures == 0 and figures:
            label = QLabel("Could not render any visualizations")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: orange; font-style: italic; margin: 20px;")
            label.setPixmap(load_bootstrap_icon("exclamation-triangle", color="#FFA500", size=32).pixmap(32, 32))
            self.viz_layout.addWidget(label)

    def parse_test_result(self, test_result):
        """
        Parse statistical test results into a standardized format.
        
        Args:
            test_result: The test result object
            
        Returns:
            dict: A structured dictionary with all test results
        """
        if test_result is None:
            return {}
            
        if isinstance(test_result, dict):
            parsed_result = test_result.copy()
        else:
            try:
                parsed_result = dict(test_result)
            except (TypeError, ValueError):
                parsed_result = getattr(test_result, '__dict__', {})
        
        if 'figures' in parsed_result:
            figures = parsed_result['figures']
            if isinstance(figures, dict):
                parsed_result['figures'] = {
                    name: fig for name, fig in figures.items() 
                    if fig is not None
                }
        
        for key, value in list(parsed_result.items()):
            if isinstance(value, dict) and 'figures' in value:
                nested_figures = value.pop('figures', {})
                if 'figures' not in parsed_result:
                    parsed_result['figures'] = {}
                for fig_name, fig in nested_figures.items():
                    if fig is not None:
                        parsed_result['figures'][f"{key}_{fig_name}"] = fig
        
        if 'alpha' not in parsed_result:
            parsed_result['alpha'] = 0.05
        
        if 'significant' not in parsed_result and 'p_value' in parsed_result:
            try:
                p_value = float(parsed_result['p_value'])
                alpha = float(parsed_result.get('alpha', 0.05))
                if not (pd.isna(p_value) or pd.isna(alpha)):
                    parsed_result['significant'] = p_value < alpha
            except (ValueError, TypeError):
                pass
        
        return parsed_result

    def parse_assumption_check(self, category, result_info):
        """
        Parse assumption check results.
        
        Args:
            category (str): The category of the assumption.
            result_info (dict): The raw assumption check information.
            
        Returns:
            dict: A structured dictionary with formatted assumption check results.
        """
        formatted_result = {
            'result': None,
            'details': {},
            'statistics': {},
            'warnings': [],
            'recommendation': '',
            'figures': {}
        }
        
        result_value = result_info.get('result')
        if result_value is not None:
            if isinstance(result_value, bool):
                formatted_result['result'] = 'PASSED' if result_value else 'FAILED'
            elif isinstance(result_value, str):
                formatted_result['result'] = result_value.upper()
            else:
                formatted_result['result'] = str(result_value).upper()
        else:
            formatted_result['result'] = 'UNKNOWN'
        
        for key in ['details', 'statistics', 'warnings', 'recommendation', 'figures']:
            if key in result_info and result_info[key]:
                formatted_result[key] = result_info[key]
        
        return formatted_result

    def populate_details_text(self, result, parsed_result):
        """Populate detailed text display with formatted test results."""
        html = f"<h2>Test Results for: {result.outcome_name}</h2>"
        html += f"<p><b>Statistical Test:</b> {parsed_result.get('test_name', 'Unknown Test')}</p>"
        html += f"<p><b>Dataset:</b> {result.dataset_name}</p>"
        
        if 'formula' in parsed_result:
            html += f"<p><b>Formula:</b> {parsed_result['formula']}</p>"
        
        if 'significant' in parsed_result:
            significant = parsed_result['significant']
            result_color = "green" if significant else "red"
            html += f"<p><b>Result:</b> <span style='color:{result_color};'>"
            html += "SIGNIFICANT" if significant else "NOT SIGNIFICANT"
            html += f"</span></p>"
        
        if 'p_value' in parsed_result:
            html += f"<p><b>p-value:</b> {parsed_result['p_value']:.4f}</p>"
        
        if 'statistic' in parsed_result:
            html += f"<p><b>Test Statistic:</b> {parsed_result['statistic']:.4f}</p>"
        
        if 'df' in parsed_result:
            html += f"<p><b>Degrees of Freedom:</b> {parsed_result['df']}</p>"
        elif 'df_model' in parsed_result and 'df_residual' in parsed_result:
            html += f"<p><b>Degrees of Freedom (Model):</b> {parsed_result['df_model']}</p>"
            html += f"<p><b>Degrees of Freedom (Residual):</b> {parsed_result['df_residual']}</p>"
        
        if 'r_squared' in parsed_result:
            html += "<h3>Model Fit</h3>"
            r_squared = parsed_result['r_squared']
            
            if isinstance(r_squared, dict):
                html += "<table border='1' cellspacing='0' cellpadding='3' style='margin-bottom:10px;'>"
                for key, value in r_squared.items():
                    html += f"<tr><td><b>{key.replace('_', ' ').title()}</b></td><td>{value:.4f}</td></tr>"
                html += "</table>"
            else:
                html += f"<p><b>R-squared:</b> {r_squared:.4f}</p>"
            
            if 'adj_r_squared' in parsed_result:
                html += f"<p><b>Adjusted R-squared:</b> {parsed_result['adj_r_squared']:.4f}</p>"
        
        if 'f_statistic' in parsed_result:
            html += f"<p><b>F-statistic:</b> {parsed_result['f_statistic']:.4f}</p>"
        
        if 'confidence_interval' in parsed_result:
            ci = parsed_result['confidence_interval']
            if isinstance(ci, dict) and 'lower' in ci and 'upper' in ci:
                html += f"<p><b>Confidence Interval:</b> ({ci['lower']:.4f}, {ci['upper']:.4f})</p>"
        
        if 'effect_size' in parsed_result:
            html += "<h3>Effect Size</h3>"
            effect_size = parsed_result['effect_size']
            
            if isinstance(effect_size, dict):
                html += "<table border='1' cellspacing='0' cellpadding='3' style='margin-bottom:10px;'>"
                for key, value in effect_size.items():
                    if key != 'interpretation':
                        if isinstance(value, float):
                            html += f"<tr><td><b>{key.replace('_', ' ').title()}</b></td><td>{value:.4f}</td></tr>"
                        else:
                            html += f"<tr><td><b>{key.replace('_', ' ').title()}</b></td><td>{value}</td></tr>"
                html += "</table>"
                
                if 'interpretation' in effect_size:
                    html += f"<p><b>Interpretation:</b> {effect_size['interpretation']}</p>"
            else:
                html += f"<p>{effect_size:.4f}</p>"
        
        if 'descriptive_stats' in parsed_result:
            html += "<h3>Descriptive Statistics</h3>"
            html += "<table border='1' cellspacing='0' cellpadding='3' style='margin-bottom:10px;'>"
            for key, value in parsed_result['descriptive_stats'].items():
                if isinstance(value, float):
                    html += f"<tr><td><b>{key.replace('_', ' ').title()}</b></td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td><b>{key.replace('_', ' ').title()}</b></td><td>{value}</td></tr>"
            html += "</table>"
        
        if 'prediction_stats' in parsed_result:
            html += "<h3>Prediction Statistics</h3>"
            html += "<table border='1' cellspacing='0' cellpadding='3' style='margin-bottom:10px;'>"
            for key, value in parsed_result['prediction_stats'].items():
                if isinstance(value, float):
                    html += f"<tr><td><b>{key.upper()}</b></td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td><b>{key.upper()}</b></td><td>{value}</td></tr>"
            html += "</table>"
        
        if 'coefficients_structured' in parsed_result:
            html += "<h3>Coefficients</h3>"
            html += "<table cellspacing='0' cellpadding='3' style='margin-bottom:10px;'>"
            html += "<tr><th>Variable</th><th>Coefficient</th><th>Std. Error</th><th>t/z value</th><th>p-value</th><th>Significant</th><th>95% CI</th></tr>"
            
            for coef in parsed_result['coefficients_structured']:
                sig_color = "green" if coef.get('significant', False) else "black"
                html += f"<tr>"
                html += f"<td>{coef.get('name', '')}</td>"
                html += f"<td>{coef.get('coef', 0):.4f}</td>"
                html += f"<td>{coef.get('std_err', 0):.4f}</td>"
                
                if 't_value' in coef:
                    html += f"<td>{coef.get('t_value', 0):.4f}</td>"
                else:
                    html += f"<td>{coef.get('z_value', 0):.4f}</td>"
                    
                html += f"<td>{coef.get('p_value', 1):.4f}</td>"
                html += f"<td style='color:{sig_color};'>{coef.get('significant', False)}</td>"
                html += f"<td>({coef.get('ci_lower', 0):.4f}, {coef.get('ci_upper', 0):.4f})</td>"
                html += f"</tr>"
            html += "</table>"
        
        if 'aic' in parsed_result or 'bic' in parsed_result:
            html += "<h3>Information Criteria</h3>"
            html += "<table border='1' cellspacing='0' cellpadding='3' style='margin-bottom:10px;'>"
            if 'aic' in parsed_result:
                html += f"<tr><td><b>AIC</b></td><td>{parsed_result['aic']:.4f}</td></tr>"
            if 'bic' in parsed_result:
                html += f"<tr><td><b>BIC</b></td><td>{parsed_result['bic']:.4f}</td></tr>"
            html += "</table>"
        
        if 'anova_table' in parsed_result:
            html += "<h3>ANOVA Table</h3>"
            anova_table = parsed_result['anova_table']
            if isinstance(anova_table, pd.DataFrame):
                html += anova_table.to_html(float_format=lambda x: f"{x:.4f}")
            else:
                html += f"<p>{str(anova_table)}</p>"
        
        if 'summary' in parsed_result:
            html += "<h3>Summary</h3>"
            html += f"<pre>{parsed_result['summary']}</pre>"
        
        if 'warnings' in parsed_result and parsed_result['warnings']:
            html += "<h3>Warnings</h3>"
            html += "<ul style='margin-bottom:10px;'>"
            for warning in parsed_result['warnings']:
                html += f"<li style='color:orange;'>{warning}</li>"
            html += "</ul>"
        
        if 'details' in parsed_result and parsed_result['details']:
            html += "<h3>Additional Details</h3>"
            html += f"<p>{parsed_result['details']}</p>"
        
        self.details_scroll.setHtml(html)

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
    
    def set_theme(self, is_dark_mode):
        """Update the widget's theme (dark/light mode)."""
        self.is_dark_mode = is_dark_mode
        if hasattr(self, 'outcomes_combo') and self.outcomes_combo.currentIndex() >= 0:
            self.on_outcome_selected(self.outcomes_combo.currentIndex())

    def matplotlib_to_qt(self, figure):
        """Convert a matplotlib figure to a Qt widget (for inline display)."""
        self.apply_theme_to_figure(figure)
        canvas = FigureCanvas(figure)
        # Inline display may have fixed height; modal display will override sizing
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        canvas.setMinimumHeight(400)
        canvas.setMaximumHeight(500)
        figure.tight_layout()
        canvas.draw()
        return canvas
        
    def apply_theme_to_figure(self, figure):
        """Apply the current theme to a matplotlib figure."""
        if self.is_dark_mode:
            figure.patch.set_facecolor('#2D2D2D')
            text_color = 'white'
            grid_color = '#555555'
            spine_color = '#888888'
        else:
            figure.patch.set_facecolor('#FFFFFF')
            text_color = 'black'
            grid_color = '#CCCCCC'
            spine_color = '#888888'
        
        for ax in figure.get_axes():
            ax.set_facecolor(figure.get_facecolor())
            if ax.get_title():
                ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(figure.get_facecolor())
                legend.get_frame().set_edgecolor(spine_color)
                for text in legend.get_texts():
                    text.set_color(text_color)
            for text in ax.texts:
                text.set_color(text_color)
        return figure

    def create_figure_widget(self, figure, title=None):
        """Create a widget containing a figure (matplotlib or SVG) with an optional title."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        if title:
            title_container = QWidget()
            title_layout = QHBoxLayout(title_container)
            title_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add chart icon to title
            icon_label = QLabel()
            icon_label.setPixmap(load_bootstrap_icon("graph-up", size=16).pixmap(16, 16))
            title_layout.addWidget(icon_label)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            title_layout.addWidget(title_label)
            title_layout.addStretch()
            
            # Add maximize button with icon
            maximize_btn = QPushButton()
            maximize_btn.setIcon(load_bootstrap_icon("arrows-fullscreen", size=14))
            maximize_btn.setToolTip("View in fullscreen")
            maximize_btn.setFixedSize(24, 24)
            maximize_btn.clicked.connect(lambda: self.show_plot_modal(figure, title))
            title_layout.addWidget(maximize_btn)
            
            # Add analyze button with icon
            analyze_btn = QPushButton()
            analyze_btn.setIcon(load_bootstrap_icon("chat-text", size=14))
            analyze_btn.setToolTip("Analyze with Claude")
            analyze_btn.setFixedSize(24, 24)
            analyze_btn.clicked.connect(lambda: self.analyze_figure_with_llm(figure, title))
            title_layout.addWidget(analyze_btn)
            
            layout.addWidget(title_container)

        if isinstance(figure, str):
            try:
                svg_content = self._get_svg_content(figure)
                if svg_content:
                    if self.is_dark_mode:
                        svg_content = self._apply_dark_theme_to_svg(svg_content)
                    else:
                        svg_content = self._apply_light_theme_to_svg(svg_content)
                    
                    svg_widget = QSvgWidget()
                    svg_widget.renderer().load(QByteArray(svg_content.encode('utf-8')))
                    svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    aspect_widget = SVGAspectRatioWidget(svg_widget)
                    layout.addWidget(aspect_widget)
                else:
                    self._add_placeholder_label(layout, "Invalid SVG content")
            except Exception as e:
                self._add_error_label(layout, e)
        elif isinstance(figure, Figure):
            try:
                canvas = self.matplotlib_to_qt(figure)
                layout.addWidget(canvas)
            except Exception as e:
                error_label = QLabel(f"Error displaying plot: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_label.setStyleSheet("color: red; font-style: italic;")
                layout.addWidget(error_label)
        elif hasattr(figure, 'figure') and isinstance(figure.figure, Figure):
            try:
                canvas = self.matplotlib_to_qt(figure.figure)
                layout.addWidget(canvas)
            except Exception as e:
                error_label = QLabel(f"Error displaying plot: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_label.setStyleSheet("color: red; font-style: italic;")
                layout.addWidget(error_label)
        else:
            placeholder = QLabel(f"Plot not available (unsupported format: {type(figure).__name__})")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(placeholder)
        
        separator = QWidget()
        separator.setFixedHeight(20)
        layout.addWidget(separator)
        
        return widget

    def _get_svg_content(self, figure):
        """Extract SVG content from string or base64."""
        if figure.startswith('PD94bWwgdm') or figure.startswith('PHN2Zw'):
            try:
                padding_needed = len(figure) % 4
                if padding_needed:
                    figure += '=' * (4 - padding_needed)
                return base64.b64decode(figure).decode('utf-8')
            except:
                pass
        if figure.strip().lower().startswith('<svg') or '<!doctype svg' in figure.lower():
            return figure
        return None

    def _apply_dark_theme_to_svg(self, svg_content):
        """Apply dark theme colors to SVG content."""
        replacements = {
            'white': '#2D2D2D',
            '#fff': '#2D2D2D',
            '#ffffff': '#2D2D2D',
            'black': '#FFFFFF',
            '#000': '#FFFFFF',
            '#000000': '#FFFFFF'
        }
        for old_color, new_color in replacements.items():
            svg_content = svg_content.replace(f'fill="{old_color}"', f'fill="{new_color}"')
            svg_content = svg_content.replace(f'stroke="{old_color}"', f'stroke="{new_color}"')
        return svg_content

    def _apply_light_theme_to_svg(self, svg_content):
        """Apply light theme colors to SVG content."""
        replacements = {
            '#2D2D2D': 'white',
            'rgb(45,45,45)': 'white',
            '#FFFFFF': 'black',
            '#ffffff': 'black'
        }
        for old_color, new_color in replacements.items():
            svg_content = svg_content.replace(f'fill="{old_color}"', f'fill="{new_color}"')
            svg_content = svg_content.replace(f'stroke="{old_color}"', f'stroke="{new_color}"')
        return svg_content

    def _add_placeholder_label(self, layout, text):
        """Add a placeholder label to the layout."""
        placeholder = QLabel(text)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(placeholder)

    def _add_error_label(self, layout, exception):
        """Add an error label to the layout."""
        error_label = QLabel(f"Error processing visualization: {str(exception)}")
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        error_label.setStyleSheet("color: red; font-style: italic;")
        layout.addWidget(error_label)

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
                svg_content = self._get_svg_content(figure)
                if svg_content:
                    if self.is_dark_mode:
                        svg_content = self._apply_dark_theme_to_svg(svg_content)
                    else:
                        svg_content = self._apply_light_theme_to_svg(svg_content)
                    
                    svg_widget = QSvgWidget()
                    svg_widget.renderer().load(QByteArray(svg_content.encode('utf-8')))
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
            self.apply_theme_to_figure(figure)
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            figure.tight_layout()
            canvas.draw()
            layout.addWidget(canvas)
        
        elif hasattr(figure, 'figure') and isinstance(figure.figure, Figure):
            self.apply_theme_to_figure(figure.figure)
            canvas = FigureCanvas(figure.figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            figure.figure.tight_layout()
            canvas.draw()
            layout.addWidget(canvas)
        
        else:
            placeholder = QLabel(f"Unsupported plot format: {type(figure).__name__}")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder)
        
        dialog.setLayout(layout)
        dialog.showMaximized()
        dialog.exec()

    def show_tree_context_menu(self, position):
        """Show context menu for tree items."""
        item = self.summary_tree.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        analyze_action = menu.addAction(load_bootstrap_icon("chat-text"), "Analyze with LLM")
        analyze_action.triggered.connect(lambda: self.analyze_tree_item_with_llm(item))
        menu.exec(QCursor.pos())
    
    def analyze_tree_item_with_llm(self, item):
        """Analyze a specific tree item with Claude."""
        # Extract data from the selected item and its children
        data = self.extract_item_data(item)
        
        # Create prompt for Claude
        prompt = self._create_item_analysis_prompt(item, data)
        
        # Show loading indicator
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Use the generic LLM call function
            from qt_sections.llm_manager import llm_config
            response = call_llm_sync(prompt, model=llm_config.default_text_model)
            self.show_llm_response_dialog("Analysis Results", response)
        except Exception as e:
            self.show_llm_response_dialog("Error", f"An error occurred: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def extract_item_data(self, item):
        """Extract data hierarchy from a tree item."""
        data = {}
        
        # Get data from the current item
        data["name"] = item.text(0)
        data["value"] = item.text(1)
        
        # Recursively get data from children
        if item.childCount() > 0:
            data["children"] = []
            for i in range(item.childCount()):
                child_data = self.extract_item_data(item.child(i))
                data["children"].append(child_data)
        
        return data
    
    def _create_item_analysis_prompt(self, item, data):
        """Create a prompt for Claude to analyze a tree item."""
        # Get the current outcome name and any other relevant context
        current_outcome = ""
        current_test = "Unknown test"
        
        if hasattr(self, 'outcomes_combo') and self.outcomes_combo.currentIndex() >= 0:
            result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
            if result:
                if hasattr(result, 'outcome_name'):
                    current_outcome = result.outcome_name
                if hasattr(result, 'statistical_test_name') and result.statistical_test_name:
                    current_test = result.statistical_test_name
        
        prompt = f"""As a statistical expert, provide a concise analysis of this statistical data point:

Item: {data['name']}
Test: {current_test}
Outcome being analyzed: {current_outcome}

Here is the data structure:
{self._format_data_for_prompt(data)}

In 3-4 sentences, explain:
1. What this parameter represents in the context of the statistical test
2. The significance of the value shown
3. How this result should be interpreted
4. Any important implications this has for the outcome

Keep your analysis brief, focused, and in simple language for someone with basic statistical knowledge."""

        return prompt
    
    def _format_data_for_prompt(self, data, indent=0):
        """Format the extracted data in a readable way for the prompt."""
        result = " " * indent + f"{data['name']}: {data['value']}\n"
        
        if "children" in data and data["children"]:
            for child in data["children"]:
                result += self._format_data_for_prompt(child, indent + 2)
        
        return result
    
    def analyze_results_with_llm(self):
        """Analyze both the statistical results and figures with Claude."""
        if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
            self.show_llm_response_dialog("Error", "No outcome selected for analysis.")
            return
        
        result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
        if not result or not hasattr(result, 'test_results'):
            self.show_llm_response_dialog("Error", "No test results available for analysis.")
            return
        
        # Parse test results and extract figures
        parsed_result = self.parse_test_result(result.test_results)
        figures = parsed_result.get('figures', {})
        
        if not figures:
            # No figures, just do a text analysis
            self._analyze_text_results_with_llm(result, parsed_result)
            return
        
        # Show loading indicator
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Create a worker thread to handle the async operation
            from PyQt6.QtCore import QThread, pyqtSignal
            
            class AsyncWorker(QThread):
                finished = pyqtSignal(str)
                error = pyqtSignal(str)
                
                def __init__(self, coro_func, *args, **kwargs):
                    super().__init__()
                    self.coro_func = coro_func
                    self.args = args
                    self.kwargs = kwargs
                
                def run(self):
                    try:
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        # Run the coroutine and get the result
                        result = loop.run_until_complete(self.coro_func(*self.args, **self.kwargs))
                        loop.close()
                        self.finished.emit(result)
                    except Exception as e:
                        self.error.emit(str(e))
            
            # Create and configure the worker
            self.worker = AsyncWorker(self._analyze_results_with_figures_async, result, parsed_result, figures)
            
            # Connect signals
            self.worker.finished.connect(lambda response: 
                self.show_llm_response_dialog("Statistical Analysis with Visuals", response))
            self.worker.error.connect(lambda error_msg: 
                self.show_llm_response_dialog("Error", f"An error occurred: {error_msg}"))
            self.worker.finished.connect(lambda _: QApplication.restoreOverrideCursor())
            self.worker.error.connect(lambda _: QApplication.restoreOverrideCursor())
            
            # Start the worker
            self.worker.start()
            
        except Exception as e:
            self.show_llm_response_dialog("Error", f"An error setting up analysis: {str(e)}")
            QApplication.restoreOverrideCursor()
            
    def analyze_figure_with_llm(self, figure, title=None):
        """Analyze a single figure with Claude."""
        if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
            self.show_llm_response_dialog("Error", "No outcome selected for analysis.")
            return
        
        result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
        if not result or not hasattr(result, 'test_results'):
            self.show_llm_response_dialog("Error", "No test results available for analysis.")
            return
        
        # Parse test results for context
        parsed_result = self.parse_test_result(result.test_results)
        
        # Create prompt specific to this figure
        prompt = self._create_single_figure_analysis_prompt(result, parsed_result, title)
        
        # Show loading indicator
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Create a worker thread to handle the async operation
            from PyQt6.QtCore import QThread, pyqtSignal
            from qt_sections.llm_manager import llm_config
            
            class AsyncWorker(QThread):
                finished = pyqtSignal(str)
                error = pyqtSignal(str)
                
                def __init__(self, prompt, figure):
                    super().__init__()
                    self.prompt = prompt
                    self.figure = figure
                
                def run(self):
                    try:
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        # Use the generic vision function instead of Claude-specific
                        result = loop.run_until_complete(call_llm_vision_async(self.prompt, self.figure, model=llm_config.default_vision_model))
                        loop.close()
                        self.finished.emit(result)
                    except Exception as e:
                        self.error.emit(str(e))
            
            # Create and configure the worker
            self.fig_worker = AsyncWorker(prompt, figure)
            
            # Connect signals
            self.fig_worker.finished.connect(lambda response: 
                self.show_llm_response_dialog(f"Analysis of {title}", response))
            self.fig_worker.error.connect(lambda error_msg: 
                self.show_llm_response_dialog("Error", f"An error occurred: {error_msg}"))
            self.fig_worker.finished.connect(lambda _: QApplication.restoreOverrideCursor())
            self.fig_worker.error.connect(lambda _: QApplication.restoreOverrideCursor())
            
            # Start the worker
            self.fig_worker.start()
            
        except Exception as e:
            self.show_llm_response_dialog("Error", f"An error setting up analysis: {str(e)}")
            QApplication.restoreOverrideCursor()

    def _create_single_figure_analysis_prompt(self, result, parsed_result, figure_name):
        """Create a prompt for Claude to analyze a single figure."""
        outcome_name = result.outcome_name
        dataset_name = result.dataset_name
        
        # Get study design and variables information
        variables_info = ""
        if hasattr(result, 'variables') and result.variables:
            if result.variables.get('outcome'):
                variables_info += f"Outcome variable: {result.variables.get('outcome')}\n"
            if result.variables.get('group'):
                variables_info += f"Group/treatment variable: {result.variables.get('group')}\n"
            if result.variables.get('covariates'):
                variables_info += f"Covariates: {', '.join(result.variables.get('covariates') or [])}\n"
            if result.variables.get('predictors'):
                variables_info += f"Predictors: {', '.join(result.variables.get('predictors') or [])}\n"
                
        # Use the actual test name from result if available
        test_name = parsed_result.get('test_name', 'Unknown test')
        if hasattr(result, 'statistical_test_name') and result.statistical_test_name:
            test_name = result.statistical_test_name
        
        prompt = f"""As a statistical expert, provide a concise analysis of this visualization:

Figure: {figure_name}
Test: {test_name}
Outcome: {outcome_name}
Dataset: {dataset_name}
{variables_info}

In 4-5 lines, please provide:
1. What this visualization shows about the data
2. How it relates to the statistical test results
3. Key patterns or notable features visible in the plot
4. What insight it adds to understanding the outcome

Focus only on the most important aspects visible in this visualization.
Explain in clear language someone with basic statistical knowledge would understand."""

        return prompt

    def _analyze_text_results_with_llm(self, result, parsed_result):
        """Analyze just the statistical results text with Claude."""
        # Create a prompt that focuses on the statistical results
        prompt = self._create_text_analysis_prompt(result, parsed_result)
        
        # Show loading indicator
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            from qt_sections.llm_manager import llm_config
            response = call_llm_sync(prompt, model=llm_config.default_text_model)
            self.show_llm_response_dialog("Statistical Analysis", response)
        except Exception as e:
            self.show_llm_response_dialog("Error", f"An error occurred: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def _create_text_analysis_prompt(self, result, parsed_result):
        """Create a prompt for Claude to analyze statistical text results."""
        test_name = parsed_result.get('test_name', 'Unknown test')
        
        # Use the actual test name from result if available
        if hasattr(result, 'statistical_test_name') and result.statistical_test_name:
            test_name = result.statistical_test_name
        
        # Get variables information (skip dataset_name and outcome_name)
        variables_info = ""
        
        # Extract predictor variables and other test metadata
        if hasattr(result, 'variables') and result.variables:
            # Extract only the specific variables used in the analysis
            if result.variables.get('predictors'):
                variables_info += f"Predictors: {', '.join(result.variables.get('predictors') or [])}\n"
            if result.variables.get('covariates'):
                variables_info += f"Covariates: {', '.join(result.variables.get('covariates') or [])}\n"
            
        # Format key statistics - exclude metadata fields
        stats_text = ""
        excluded_keys = ['figures', 'test_name', 'outcome_name', 'dataset_name', 'alpha']
        
        for key, value in parsed_result.items():
            if key not in excluded_keys:
                if isinstance(value, dict):
                    stats_text += f"\n{key}:\n"
                    for k, v in value.items():
                        stats_text += f"  {k}: {v}\n"
                elif not isinstance(value, (list, tuple)):
                    stats_text += f"\n{key}: {value}"
        
        # Format coefficients if available
        coef_text = ""
        if 'coefficients_structured' in parsed_result:
            coef_text += "\nCoefficients:\n"
            for coef in parsed_result['coefficients_structured']:
                coef_text += f"- {coef.get('name', 'Unknown')}: {coef.get('coef', 'N/A')}, p-value: {coef.get('p_value', 'N/A')}, significant: {coef.get('significant', 'N/A')}\n"
        
        prompt = f"""As a statistical expert, analyze the following statistical test results concisely:

Test: {test_name}
{variables_info}

Key Statistics:
{stats_text}

{coef_text}

Provide a BRIEF 4-5 line interpretation including:
1. The key finding and statistical significance
2. The practical importance of the findings
3. Any important limitations or caveats
4. A simple recommendation based on these results

Focus on analyzing the statistical results themselves rather than describing the study metadata.
Keep your analysis concise and focused on the most important aspects of these results.
Use clear language that could be understood by someone with basic statistical knowledge."""

        return prompt
    
    async def _analyze_results_with_figures_async(self, result, parsed_result, figures):
        """Analyze both statistical results and figures with Claude asynchronously."""
        # Process figures for Claude
        figure_objects = []
        for fig_name, fig in figures.items():
            if fig is not None:
                figure_objects.append(fig)
        
        # Create a prompt that includes both statistical results and figures
        prompt = self._create_figures_analysis_prompt(result, parsed_result, figures.keys())
        
        # Call the generic LLM vision function
        from qt_sections.llm_manager import llm_config
        # Use call_llm_vision_async with the first figure only
        # Multiple image support varies by model and should be handled in the client.py
        return await call_llm_vision_async(prompt, figure_objects[0] if figure_objects else None, model=llm_config.default_vision_model)
    
    def _create_figures_analysis_prompt(self, result, parsed_result, figure_names):
        """Create a prompt for Claude to analyze both statistics and figures."""
        # First get the text analysis prompt
        base_prompt = self._create_text_analysis_prompt(result, parsed_result)
        
        # Add figure-specific instructions
        figure_prompt = f"""
I'm also providing {len(figure_names)} visualizations related to these results.

Please also analyze these visualizations in relation to the statistical results:
1. How the visualizations support or complement the numerical results
2. Any key patterns, trends or outliers visible in the plots
3. How these visuals help interpret the findings

Keep your analysis of the visualizations brief and focused on the most important insights.
Your total response should be about 6-8 lines at most, focusing on statistical interpretation rather than metadata descriptions.
"""
        
        return base_prompt + figure_prompt
    
    def show_llm_response_dialog(self, title, content):
        """Show Claude's response in a dialog box."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Create a QTextEdit to display the response
        response_text = QTextEdit()
        response_text.setReadOnly(True)
        response_text.setMarkdown(content)
        response_text.setStyleSheet("font-size: 12pt;")
        layout.addWidget(response_text)
        
        # Button layout
        btn_layout = QHBoxLayout()
        
        # Add a save button
        save_btn = QPushButton("Save Analysis")
        save_btn.clicked.connect(lambda: self.save_llm_analysis(title, content))
        save_btn.setIcon(load_bootstrap_icon("save"))
        btn_layout.addWidget(save_btn)
        
        # Add a copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(content))
        btn_layout.addWidget(copy_btn)
        
        btn_layout.addStretch()
        
        # Add a close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def save_llm_analysis(self, analysis_type, content):
        """Save the LLM-generated analysis back to the study metadata."""
        if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
            self.show_llm_response_dialog("Error", "No active study or outcome selected.")
            return
            
        result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
        if not result or not hasattr(result, 'outcome_name'):
            self.show_llm_response_dialog("Error", "No valid outcome selected.")
            return
            
        # Check if we have a studies manager to save to
        if not self.studies_manager:
            self.show_llm_response_dialog("Error", "No studies manager available to save the analysis.")
            return
            
        try:
            # Get the current active study
            study = self.studies_manager.get_active_study()
            if not study:
                self.show_llm_response_dialog("Error", "No active study to save the analysis to.")
                return
                
            # Format the analysis metadata
            timestamp = datetime.now().isoformat()
            analysis_metadata = {
                "type": analysis_type,
                "outcome_name": result.outcome_name,
                "dataset_name": result.dataset_name if hasattr(result, 'dataset_name') else "Unknown",
                "test_name": result.statistical_test_name if hasattr(result, 'statistical_test_name') else "Unknown",
                "timestamp": timestamp,
                "content": content
            }
            
            # Save the analysis to the study metadata
            # Initialize llm_analyses if it doesn't exist
            if not hasattr(study, 'llm_analyses'):
                study.llm_analyses = []
            
            # Add the new analysis
            study.llm_analyses.append(analysis_metadata)
            
            # Update the study's timestamp
            study.updated_at = datetime.now().isoformat()
            
            self.show_llm_response_dialog("Success", "Analysis saved successfully to study metadata.")
        except Exception as e:
            self.show_llm_response_dialog("Error", f"Failed to save analysis: {str(e)}")

    def analyze_all_tree_items_with_llm(self):
        """Analyze all top-level tree items in the summary tree with Claude."""
        if not hasattr(self, 'summary_tree') or self.summary_tree.topLevelItemCount() == 0:
            self.show_llm_response_dialog("Error", "No test results available for analysis.")
            return
        
        if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
            self.show_llm_response_dialog("Error", "No outcome selected for analysis.")
            return
        
        result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
        if not result:
            self.show_llm_response_dialog("Error", "No result data available.")
            return
        
        # Collect all top-level items for analysis
        items = []
        for i in range(self.summary_tree.topLevelItemCount()):
            items.append(self.summary_tree.topLevelItem(i))
        
        if not items:
            self.show_llm_response_dialog("Error", "No items to analyze.")
            return
        
        # Show the status dialog
        self.show_analysis_status_dialog(items, [], "tree")
    
    def analyze_all_figures_with_llm(self):
        """Analyze all figures with Claude."""
        if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
            self.show_llm_response_dialog("Error", "No outcome selected for analysis.")
            return
        
        result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
        if not result or not hasattr(result, 'test_results'):
            self.show_llm_response_dialog("Error", "No test results available for analysis.")
            return
        
        # Parse test results and extract figures
        parsed_result = self.parse_test_result(result.test_results)
        figures = parsed_result.get('figures', {})
        
        if not figures:
            self.show_llm_response_dialog("Error", "No figures available for analysis.")
            return
        
        # Convert to a list of (title, figure) tuples
        figure_items = []
        for fig_name, fig in figures.items():
            if fig is not None:
                title = fig_name.replace('_', ' ').title()
                figure_items.append((title, fig))
        
        if not figure_items:
            self.show_llm_response_dialog("Error", "No valid figures available for analysis.")
            return
        
        # Show the status dialog
        self.show_analysis_status_dialog([], figure_items, "figures")
    
    def show_analysis_status_dialog(self, tree_items, figures, mode):
        """Show a dialog with status updates for all analyses."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis Progress")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        
        # Header with icon
        header_layout = QHBoxLayout()
        header_icon = QLabel()
        header_icon.setPixmap(load_bootstrap_icon("cpu", size=24).pixmap(24, 24))
        header_layout.addWidget(header_icon)
        
        header_label = QLabel("Analysis Progress")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Status message
        status_label = QLabel("Running analyses... Please wait.")
        status_label.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(status_label)
        
        # Create scroll area for the grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Container for the grid
        grid_container = QWidget()
        grid_layout = QVBoxLayout(grid_container)
        grid_layout.setSpacing(15)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add items to analyze based on mode
        tasks = []
        
        if mode == "tree" or mode == "both":
            tree_group = QGroupBox("Test Results")
            tree_group_layout = QVBoxLayout(tree_group)
            
            for item in tree_items:
                task_widget = self.create_task_widget(item.text(0), "tree", item)
                tree_group_layout.addWidget(task_widget)
                tasks.append(task_widget)
            
            grid_layout.addWidget(tree_group)
        
        if mode == "figures" or mode == "both":
            figures_group = QGroupBox("Visualizations")
            figures_group_layout = QVBoxLayout(figures_group)
            
            for title, fig in figures:
                task_widget = self.create_task_widget(title, "figure", fig)
                figures_group_layout.addWidget(task_widget)
                tasks.append(task_widget)
            
            grid_layout.addWidget(figures_group)
        
        scroll_area.setWidget(grid_container)
        layout.addWidget(scroll_area, 1)  # Give it stretch factor
        
        # Progress indicator
        progress_layout = QHBoxLayout()
        progress_label = QLabel("0 of 0 completed")
        progress_layout.addWidget(progress_label)
        progress_layout.addStretch()
        layout.addLayout(progress_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Add "Save All" button
        save_all_button = QPushButton("Save All Results")
        save_all_button.setIcon(load_bootstrap_icon("save"))
        save_all_button.clicked.connect(lambda: self.save_all_analysis_results(tasks))
        save_all_button.setEnabled(False)  # Initially disabled until analyses complete
        button_layout.addWidget(save_all_button)
        
        # Add spacer
        button_layout.addStretch()
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.reject)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # Store the save button for later enabling
        dialog.save_all_button = save_all_button
        
        # Show the dialog but don't block
        dialog.setModal(False)
        dialog.show()
        
        # Start the analyses asynchronously
        if tasks:
            self.run_analysis_tasks(tasks, dialog, progress_label, status_label)

    def create_task_widget(self, title, item_type, item_data):
        """Create a widget for tracking an analysis task."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Icon based on type
        icon_label = QLabel()
        if item_type == "tree":
            icon_label.setPixmap(load_bootstrap_icon("list-check", size=16).pixmap(16, 16))
        else:
            icon_label.setPixmap(load_bootstrap_icon("bar-chart", size=16).pixmap(16, 16))
        layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label, 1)  # Give stretch
        
        # Status indicator
        status_label = QLabel("Pending")
        status_label.setStyleSheet("color: #888;")
        layout.addWidget(status_label)
        
        # View button (initially disabled)
        view_button = QPushButton("View")
        view_button.setIcon(load_bootstrap_icon("eye"))
        view_button.setEnabled(False)
        view_button.setFixedWidth(80)
        layout.addWidget(view_button)
        
        # Store data in the widget for later
        widget.item_type = item_type
        widget.item_data = item_data
        widget.title = title
        widget.status_label = status_label
        widget.view_button = view_button
        widget.response = None
        
        return widget
    
    def run_analysis_tasks(self, tasks, dialog, progress_label, status_label):
        """Run all analysis tasks asynchronously."""
        # Create a worker thread to handle the async operations
        class AnalysisWorker(QThread):
            task_completed = pyqtSignal(int, str)
            all_completed = pyqtSignal()
            error_occurred = pyqtSignal(int, str)
            
            def __init__(self, parent, tasks):
                super().__init__(parent)
                self.tasks = tasks
                self.parent = parent
            
            def run(self):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Create tasks for all analyses
                task_futures = []
                
                for i, task_widget in enumerate(self.tasks):
                    if task_widget.item_type == "tree":
                        task_coroutine = self.analyze_tree_item(i, task_widget)
                    else:
                        task_coroutine = self.analyze_figure(i, task_widget)
                    
                    task_futures.append(loop.create_task(task_coroutine))
                
                # Wait for all tasks to complete
                loop.run_until_complete(asyncio.gather(*task_futures))
                loop.close()
                
                self.all_completed.emit()
            
            async def analyze_tree_item(self, index, task_widget):
                try:
                    item = task_widget.item_data
                    data = self.parent.extract_item_data(item)
                    prompt = self.parent._create_item_analysis_prompt(item, data)
                    
                    from qt_sections.llm_manager import llm_config
                    response = await call_llm_async(prompt, model=llm_config.default_text_model)
                    
                    # Save the response and notify about completion
                    task_widget.response = response
                    self.task_completed.emit(index, response)
                    
                    # Save to studies manager
                    self.parent.save_analysis_result(task_widget.title, response)
                except Exception as e:
                    self.error_occurred.emit(index, str(e))
            
            async def analyze_figure(self, index, task_widget):
                try:
                    title = task_widget.title
                    figure = task_widget.item_data
                    
                    # Get current outcome for context
                    outcome_index = self.parent.outcomes_combo.currentIndex()
                    result = self.parent.outcomes_combo.itemData(outcome_index)
                    parsed_result = self.parent.parse_test_result(result.test_results)
                    
                    prompt = self.parent._create_single_figure_analysis_prompt(result, parsed_result, title)
                    from qt_sections.llm_manager import llm_config
                    response = await call_llm_vision_async(prompt, figure, model=llm_config.default_vision_model)
                    
                    # Save the response and notify about completion
                    task_widget.response = response
                    self.task_completed.emit(index, response)
                    
                    # Save to studies manager
                    self.parent.save_analysis_result(title, response)
                except Exception as e:
                    self.error_occurred.emit(index, str(e))
        
        # Create and set up the worker
        self.analysis_worker = AnalysisWorker(self, tasks)
        
        # Connect signals
        self.analysis_worker.task_completed.connect(
            lambda index, response: self.update_task_status(tasks, index, "Completed", response, progress_label))
        self.analysis_worker.error_occurred.connect(
            lambda index, error: self.update_task_status(tasks, index, f"Error: {error}", None, progress_label))
        self.analysis_worker.all_completed.connect(
            lambda: self.on_all_analyses_completed(status_label))
        
        # Start the worker
        self.analysis_worker.start()

    def update_task_status(self, tasks, index, status, response, progress_label):
        """Update the status of a task in the dialog."""
        if 0 <= index < len(tasks):
            task_widget = tasks[index]
            
            # Update status label
            if status.startswith("Error"):
                task_widget.status_label.setText("Failed")
                task_widget.status_label.setStyleSheet("color: red;")
            else:
                task_widget.status_label.setText("Completed")
                task_widget.status_label.setStyleSheet("color: green;")
            
            # Store response and enable view button if response is available
            if response:
                task_widget.response = response
                task_widget.view_button.setEnabled(True)
                task_widget.view_button.clicked.connect(
                    lambda checked=False, t=task_widget: 
                    self.show_llm_response_dialog(f"Analysis of {t.title}", t.response))
            
            # Update progress counter
            completed = sum(1 for t in tasks if t.status_label.text() in ["Completed", "Failed"])
            progress_label.setText(f"{completed} of {len(tasks)} completed")

    def on_all_analyses_completed(self, status_label):
        """Handle completion of all analyses."""
        status_label.setText("All analyses completed!")
        status_label.setStyleSheet("font-style: italic; color: green;")
        
        # Enable the save all button if it exists
        if hasattr(status_label.parent(), 'save_all_button'):
            status_label.parent().save_all_button.setEnabled(True)

    def save_analysis_result(self, title, content):
        """Save an analysis result to the study metadata."""
        if not self.studies_manager:
            return
        
        try:
            # Get the current active study
            study = self.studies_manager.get_active_study()
            if not study:
                return
            
            # Get the current outcome
            if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
                return
            
            result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
            if not result or not hasattr(result, 'outcome_name'):
                return
            
            # Format the analysis metadata
            timestamp = datetime.now().isoformat()
            analysis_metadata = {
                "type": f"Test Results - {title}",
                "outcome_name": result.outcome_name,
                "dataset_name": getattr(result, 'dataset_name', "Unknown"),
                "test_name": getattr(result, 'statistical_test_name', "Unknown"),
                "timestamp": timestamp,
                "content": content
            }
            
            # Save the analysis to the study metadata
            # Initialize llm_analyses if it doesn't exist
            if not hasattr(study, 'llm_analyses'):
                study.llm_analyses = []
            
            # Add the new analysis
            study.llm_analyses.append(analysis_metadata)
            
            # Update the study's timestamp
            study.updated_at = datetime.now().isoformat()
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")

    def save_all_analysis_results(self, tasks):
        """Save all analysis results to the study metadata."""
        if not self.studies_manager:
            return
        
        try:
            # Get the current active study
            study = self.studies_manager.get_active_study()
            if not study:
                return
            
            # Collect all analysis results
            results = []
            for task_widget in tasks:
                if task_widget.response:
                    results.append((task_widget.title, task_widget.response))
            
            if not results:
                self.show_llm_response_dialog("Error", "No analysis results to save.")
                return
            
            # Save each analysis result to the study metadata
            for title, content in results:
                self.save_analysis_result(title, content)
            
            self.show_llm_response_dialog("Success", "All analysis results saved successfully!")
        except Exception as e:
            self.show_llm_response_dialog("Error", f"An error occurred: {str(e)}")


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

