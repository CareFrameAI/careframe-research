from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                            QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                            QSplitter, QTabWidget, QTextEdit, QComboBox,
                            QFormLayout, QTreeWidget, QTreeWidgetItem, QScrollArea,
                            QSizePolicy, QDialog, QMenu, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QItemSelectionModel, QByteArray, QThread
from PyQt6.QtGui import QFont, QBrush, QColor, QCursor
from PyQt6.QtSvgWidgets import QSvgWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the icon loading function
from helpers.load_icon import load_bootstrap_icon

import pandas as pd
import json
import base64
import asyncio
from datetime import datetime
from llms.client import (call_claude_sync, call_claude_async, 
                       call_claude_with_image_sync, call_claude_with_image_async,
                       call_claude_with_multiple_images_async, 
                       call_llm_sync, call_llm_async, call_llm_vision_sync, call_llm_vision_async)

class AssumptionsDisplayWidget(QWidget):
    """Widget for viewing assumption checks."""
    
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
        header_label = QLabel("Assumption Checks")
        header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        header_label.setFont(font)
        # Add icon to header
        header_label.setPixmap(load_bootstrap_icon("check-circle", size=24).pixmap(24, 24))
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
        tree_header = QLabel("Assumption Results")
        tree_header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        tree_header.setPixmap(load_bootstrap_icon("list-check", size=16).pixmap(16, 16))
        tree_layout.addWidget(tree_header)
        
        self.summary_tree = QTreeWidget()
        self.summary_tree.setHeaderLabels(["Assumption Check", "Result"])
        self.summary_tree.setColumnWidth(0, 300)
        self.summary_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add context menu to the summary tree
        self.summary_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.summary_tree.customContextMenuRequested.connect(self.show_tree_context_menu)
        
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
        
        # Create a new row for analyze buttons with better spacing
        analyze_buttons_container = QWidget()
        analyze_buttons_layout = QHBoxLayout(analyze_buttons_container)
        analyze_buttons_layout.setContentsMargins(0, 5, 0, 10)
        analyze_buttons_layout.setSpacing(10)
        
        # Add button to analyze all assumption checks with LLM
        analyze_tree_btn = QPushButton("Analyze All Assumption Checks")
        analyze_tree_btn.setIcon(load_bootstrap_icon("list-check"))
        analyze_tree_btn.clicked.connect(self.analyze_all_tree_items_with_llm)
        analyze_buttons_layout.addWidget(analyze_tree_btn)
        
        # Add button to analyze all figures with LLM
        analyze_figures_btn = QPushButton("Analyze All Visualizations")
        analyze_figures_btn.setIcon(load_bootstrap_icon("bar-chart"))
        analyze_figures_btn.clicked.connect(self.analyze_all_figures_with_llm)
        analyze_buttons_layout.addWidget(analyze_figures_btn)
        
        # Add to the main layout
        viz_layout.addWidget(analyze_buttons_container)
        
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
        
        # Wrap the splitter in a content_area widget that we can show/hide
        self.content_area = QWidget()
        content_area_layout = QVBoxLayout(self.content_area)
        content_area_layout.setContentsMargins(0, 0, 0, 0)
        content_area_layout.addWidget(content_splitter)
        self.content_area.setVisible(True)  # Hide initially
        
        main_layout.addWidget(self.content_area, stretch=1)

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
        
        if hasattr(study, 'results') and study.results:
            outcomes_with_assumptions = []
            for result in study.results:
                if (hasattr(result, 'assumptions_check') and 
                    result.assumptions_check and 
                    len(result.assumptions_check) > 0):
                    outcomes_with_assumptions.append(result)
            
            if outcomes_with_assumptions:
                for result in outcomes_with_assumptions:
                    self.outcomes_combo.addItem(load_bootstrap_icon("check-circle"), result.outcome_name, result)
                self.outcomes_section.setVisible(True)
            else:
                self.outcomes_combo.addItem(load_bootstrap_icon("exclamation-triangle", color="#FFA500"), "No assumption checks available")
                self.outcomes_section.setVisible(False)
        else:
            self.outcomes_combo.addItem(load_bootstrap_icon("exclamation-triangle", color="#FFA500"), "No assumption checks available")
            self.outcomes_section.setVisible(False)
            
    def on_outcome_selected(self, index):
        """Handle selection of an outcome in the combo box."""
        if index < 0 or not self.outcomes_combo.itemData(index):
            return
        
        result = self.outcomes_combo.itemData(index)
        self.display_assumption_checks(result)
        
    def display_assumption_checks(self, result):
        """Display assumption checks for the selected outcome."""
        # Display assumption checks for the selected outcome
        if not result or not hasattr(result, 'assumptions_check') or not result.assumptions_check:
            return
        
        # Show the content area
        self.content_area.setVisible(True)
        
        # Clear existing content
        self.summary_tree.clear()
        
        # Clear visualization tab
        self.clear_visualization_tab()
        
        # Build the summary tree
        for category, check_info in result.assumptions_check.items():
            # Extract result status for the icon
            result_status = self.extract_result_status(check_info)
            
            # Create category item
            category_item = QTreeWidgetItem(self.summary_tree, [category.replace('_', ' ').title(), result_status])
            
            # Set color based on result and add icon
            if result_status == 'PASSED':
                category_item.setForeground(1, QBrush(QColor("green")))
                icon = load_bootstrap_icon("check-circle-fill", color="green") # Add PASSED icon
                category_item.setIcon(1, icon)
            elif result_status == 'FAILED':
                category_item.setForeground(1, QBrush(QColor("red")))
                icon = load_bootstrap_icon("x-circle-fill", color="red") # Add FAILED icon
                category_item.setIcon(1, icon)
            elif result_status == 'WARNING':
                category_item.setForeground(1, QBrush(QColor("orange")))
                icon = load_bootstrap_icon("exclamation-triangle-fill", color="orange") # Add WARNING icon
                category_item.setIcon(1, icon)
            else:
                category_item.setForeground(1, QBrush(QColor("gray")))
                icon = load_bootstrap_icon("question-circle-fill", color="gray") # Add UNKNOWN icon
                category_item.setIcon(1, icon)
            
            # Recursively add all check_info fields to the tree
            self.add_info_to_tree(category_item, check_info, exclude=['figures'])
            
            # Add warnings as child items if they exist
            if 'warnings' in check_info and check_info['warnings']:
                warnings_parent = QTreeWidgetItem(category_item, ["Warnings", ""])
                for warning in check_info['warnings']:
                    QTreeWidgetItem(warnings_parent, ["Warning", warning])
            
            # Only add figures that are within assumptions structure
            if 'figures' in check_info and check_info['figures']:
                for fig_name, fig in check_info['figures'].items():
                    if fig is not None:
                        try:
                            # Create a descriptive title for the figure
                            if '_' in fig_name:
                                # For nested figures (e.g., "age_scatter_plot")
                                parts = fig_name.split('_')
                                var_name = parts[0]
                                plot_type = '_'.join(parts[1:])
                                title = f"{category.replace('_', ' ').title()} - {var_name} - {plot_type.replace('_', ' ').title()}"
                            else:
                                title = f"{category.replace('_', ' ').title()} - {fig_name.replace('_', ' ').title()}"
                            
                            # Create the widget and add it to the layout
                            fig_widget = self.create_figure_widget(fig, title)
                            if fig_widget:
                                self.viz_layout.addWidget(fig_widget)
                        except Exception as e:
                            print(f"Error adding visualization: {e}")
        # Expand the tree
        self.summary_tree.expandAll()
    
    def extract_result_status(self, check_info):
        """
        Extract the result status from check_info.
        Recursively checks nested dictionaries and lists for result status.
        """
        # Handle case where check_info is directly an AssumptionResult enum
        from data.assumptions.tests import AssumptionResult
        if isinstance(check_info, AssumptionResult):
            return check_info.name
        
        # Check for 'satisfied' key first at the current level
        if isinstance(check_info, dict):
            if 'satisfied' in check_info:
                status = 'PASSED' if check_info['satisfied'] else 'FAILED'
                return status
            
            # Check for 'result' key at the current level
            if 'result' in check_info:
                if isinstance(check_info['result'], str):
                    # Handle string result values
                    result_value = check_info['result'].upper()
                    if result_value in ['PASSED', 'FAILED', 'WARNING', 'NOT_APPLICABLE']:
                        return result_value
                elif isinstance(check_info['result'], AssumptionResult):
                    # Handle enum result values
                    return check_info['result'].name
                elif hasattr(check_info['result'], 'name'):
                    # Handle other enum result values
                    result_value = check_info['result'].name
                    return result_value
            
            # Recursively check nested dictionaries and lists
            statuses = []
            
            # Check nested dictionaries
            for key, value in check_info.items():
                if isinstance(value, dict):
                    nested_status = self.extract_result_status(value)
                    if nested_status != 'UNKNOWN':
                        statuses.append(nested_status)
                elif isinstance(value, list):
                    # Check lists of dictionaries
                    for item in value:
                        if isinstance(item, dict):
                            nested_status = self.extract_result_status(item)
                            if nested_status != 'UNKNOWN':
                                statuses.append(nested_status)
                elif isinstance(value, AssumptionResult):
                    # Handle AssumptionResult values in dictionary values
                    statuses.append(value.name)
            
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

    def add_info_to_tree(self, parent_item, info_dict, exclude=None):
        """
        Recursively add dictionary items to the tree widget.
        
        Args:
            parent_item: Parent QTreeWidgetItem
            info_dict: Dictionary containing information
            exclude: List of keys to exclude
        """
        if exclude is None:
            exclude = []
            
        for key, value in info_dict.items():
            # Skip excluded keys
            if key in exclude:
                continue
                
            if isinstance(value, dict):
                # Create a branch for dictionary
                branch = QTreeWidgetItem(parent_item, [key.replace('_', ' ').title(), ""])
                # Recursively add dictionary items
                self.add_info_to_tree(branch, value, exclude)
            elif isinstance(value, list):
                # Create a branch for list
                branch = QTreeWidgetItem(parent_item, [key.replace('_', ' ').title(), ""])
                # Add list items
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        sub_branch = QTreeWidgetItem(branch, [f"Item {i+1}", ""])
                        self.add_info_to_tree(sub_branch, item, exclude)
                    else:
                        QTreeWidgetItem(branch, [f"Item {i+1}", str(item)])
            else:
                # Add a leaf for simple value
                QTreeWidgetItem(parent_item, [key.replace('_', ' ').title(), str(value)])

    def clear_visualization_tab(self):
        """Clear all widgets from the visualization tab."""
        # Remove all existing widgets from the layout if it exists
        if hasattr(self, 'viz_layout') and self.viz_layout is not None:
            while self.viz_layout.count():
                item = self.viz_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            # First-time initialization - create container and layout
            self.viz_container = QWidget()
            self.viz_layout = QVBoxLayout(self.viz_container)
            self.viz_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.viz_scroll.setWidget(self.viz_container)
    
    def set_theme(self, is_dark_mode):
        """Update the widget's theme (dark/light mode)."""
        self.is_dark_mode = is_dark_mode
        # Refresh any current visualizations if there are any
        if hasattr(self, 'outcomes_combo') and self.outcomes_combo.currentIndex() >= 0:
            self.on_outcome_selected(self.outcomes_combo.currentIndex())

    def matplotlib_to_qt(self, figure):
        """Convert a matplotlib figure to a Qt widget."""
        # Apply theme to the figure before creating canvas
        self.apply_theme_to_figure(figure)
        
        # Create a canvas for the figure
        canvas = FigureCanvas(figure)
        
        # Set size policies to prevent excessive stretching
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        canvas.setMinimumHeight(400)
        canvas.setMaximumHeight(500)
        
        # Make sure the figure is properly rendered
        figure.tight_layout()
        canvas.draw()
        
        return canvas
        
    def apply_theme_to_figure(self, figure):
        """Apply the current theme to a matplotlib figure."""
        if self.is_dark_mode:
            # Dark mode settings
            figure.patch.set_facecolor('#2D2D2D')  # Dark background
            text_color = 'white'
            grid_color = '#555555'
            spine_color = '#888888'
        else:
            # Light mode settings
            figure.patch.set_facecolor('#FFFFFF')  # Light background
            text_color = 'black'
            grid_color = '#CCCCCC'
            spine_color = '#888888'
        
        # Apply theme to all axes in the figure
        for ax in figure.get_axes():
            # Set background, text colors, and spines
            ax.set_facecolor(figure.get_facecolor())
            
            # Update title, labels, and tick colors
            if ax.get_title():
                ax.title.set_color(text_color)
            
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            
            # Update tick colors
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            
            # Update spine colors
            for spine in ax.spines.values():
                spine.set_color(spine_color)
            
            # Update grid if it's on
            if ax.get_xgridlines() or ax.get_ygridlines():
                ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Update legend if it exists
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(figure.get_facecolor())
                legend.get_frame().set_edgecolor(spine_color)
                
                # Update legend text colors
                for text in legend.get_texts():
                    text.set_color(text_color)
            
            # Update any text elements in the plot
            for text in ax.texts:
                text.set_color(text_color)
            
            # Handle special plot types - scatter plots, bar charts, etc.
            for collection in ax.collections:
                # Only adjust if using default colors and might be hard to see
                pass
        
        # Return the themed figure
        return figure

    def create_figure_widget(self, figure, title=None):
        """Create a widget containing a figure (matplotlib or SVG) with an optional title."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Set size policy for the figure widget
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Add title if provided
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
        
        # Add the figure canvas
        if figure is None:
            # Handle None figures
            placeholder = QLabel("Plot not available")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(placeholder)
        elif isinstance(figure, str):
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
            # Handle case where figure is a wrapper object with a figure attribute
            try:
                canvas = self.matplotlib_to_qt(figure.figure)
                layout.addWidget(canvas)
            except Exception as e:
                error_label = QLabel(f"Error displaying plot: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_label.setStyleSheet("color: red; font-style: italic;")
                layout.addWidget(error_label)
        else:
            # If not a matplotlib figure, show a placeholder
            placeholder = QLabel(f"Plot not available (unsupported format: {type(figure).__name__})")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(placeholder)
        
        # Add a bottom margin/separator
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
            # Use the generic LLM call function instead of Claude-specific
            from qt_sections.llm_manager import llm_config
            response = call_llm_sync(prompt, model=llm_config.default_text_model)
            self.show_llm_response_dialog("Assumption Analysis", response)
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
        
        if hasattr(self, 'outcomes_combo') and self.outcomes_combo.currentIndex() >= 0:
            result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
            if result and hasattr(result, 'outcome_name'):
                current_outcome = result.outcome_name
        
        prompt = f"""As a statistical expert, provide a concise analysis of this assumption check:

Item: {data['name']}
Result: {data['value']}
Outcome being analyzed: {current_outcome}

Here is the data structure:
{self._format_data_for_prompt(data)}

In 3-4 sentences, explain:
1. What this assumption represents in statistical analysis
2. The significance of the result shown (passed, failed, warning)
3. How this result should be interpreted
4. What implications this has for the validity of the statistical test and results

Keep your analysis brief, focused, and in simple language for someone with basic statistical knowledge."""

        return prompt
    
    def _format_data_for_prompt(self, data, indent=0):
        """Format the extracted data in a readable way for the prompt."""
        result = " " * indent + f"{data['name']}: {data['value']}\n"
        
        if "children" in data and data["children"]:
            for child in data["children"]:
                result += self._format_data_for_prompt(child, indent + 2)
        
        return result
    
    def analyze_all_tree_items_with_llm(self):
        """Analyze all top-level tree items in the summary tree with Claude."""
        if not hasattr(self, 'summary_tree') or self.summary_tree.topLevelItemCount() == 0:
            self.show_llm_response_dialog("Error", "No assumption checks available for analysis.")
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
        if not result or not hasattr(result, 'assumptions_check'):
            self.show_llm_response_dialog("Error", "No assumption checks available for analysis.")
            return
            
        # Collect all figures from assumption checks
        figures = []
        for category, check_info in result.assumptions_check.items():
            if 'figures' in check_info and check_info['figures']:
                for fig_name, fig in check_info['figures'].items():
                    if fig is not None:
                        title = f"{category.replace('_', ' ').title()} - {fig_name.replace('_', ' ').title()}"
                        figures.append((title, fig))
        
        if not figures:
            self.show_llm_response_dialog("Error", "No figures available for analysis.")
            return
            
        # Show the status dialog
        self.show_analysis_status_dialog([], figures, "figures")
        
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
            tree_group = QGroupBox("Assumption Checks")
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
                    
                    prompt = self.parent._create_single_figure_analysis_prompt(result, title)
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
                "type": f"Assumptions - {title}",
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
                "type": f"Assumptions - {analysis_type}",
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
            
            self.show_llm_response_dialog("Success", "Analysis saved successfully to study metadata.")
        except Exception as e:
            self.show_llm_response_dialog("Error", f"Failed to save analysis: {str(e)}")

    def analyze_figure_with_llm(self, figure, title=None):
        """Analyze a single figure with Claude."""
        if not hasattr(self, 'outcomes_combo') or self.outcomes_combo.currentIndex() < 0:
            self.show_llm_response_dialog("Error", "No outcome selected for analysis.")
            return
        
        result = self.outcomes_combo.itemData(self.outcomes_combo.currentIndex())
        if not result or not hasattr(result, 'assumptions_check'):
            self.show_llm_response_dialog("Error", "No assumption checks available for analysis.")
            return
        
        # Create prompt specific to this figure
        prompt = self._create_single_figure_analysis_prompt(result, title)
        
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
            
    def _create_single_figure_analysis_prompt(self, result, figure_name):
        """Create a prompt for Claude to analyze a single figure."""
        outcome_name = result.outcome_name
        dataset_name = getattr(result, 'dataset_name', "Unknown")
        
        # Get study design and variables information if available
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
        test_name = getattr(result, 'statistical_test_name', "Unknown test")
        
        prompt = f"""As a statistical expert, provide a concise analysis of this assumption check visualization:

Figure: {figure_name}
Test: {test_name}
Outcome: {outcome_name}
Dataset: {dataset_name}
{variables_info}

In 4-5 lines, please provide:
1. What this visualization shows about the assumption being checked
2. Whether the assumption appears to be met based on the visualization
3. Key patterns or notable features visible in the plot
4. What this means for the validity of the statistical test

Focus only on the most important aspects visible in this visualization.
Explain in clear language someone with basic statistical knowledge would understand."""

        return prompt

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
