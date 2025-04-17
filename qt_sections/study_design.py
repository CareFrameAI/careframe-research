from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QMessageBox, QDialog, QGroupBox, QTableWidget, QTableWidgetItem,
                             QLineEdit, QDialogButtonBox, QComboBox, QListWidget, QStackedWidget, QListWidgetItem,
                             QSplitter, QFrame, QScrollArea, QGridLayout, QSizePolicy, QTabWidget, QHeaderView, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize, QDate
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDateEdit,
                             QPushButton, QLabel, QTextEdit, QMessageBox, QDialog, QGroupBox)
from PyQt6.QtCore import Qt
import asyncio
import json
import pandas as pd
from study_model.study_model import StudyType, CFDataType, TimePoint, InterventionType, OutcomeCategory, DataCollectionMethod, BlindingType, RandomizationMethod, EligibilityOperator, AdverseEventSeverity, AdverseEventCausality
from db_ops.generate_tables import DatabaseSetup
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from helpers.load_icon import load_bootstrap_icon


class StudyDesignSection(QWidget):
    analysis_plan_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None, client_id="default_user"):
        super().__init__(parent)
        self.main = parent
        self.client_id = client_id
        # Store available datasets from the active study
        self.available_datasets = {}  # {name: dataframe}
        self.current_dataset = None
        self.current_dataset_variables = []  # List of column names
        self.db_setup = DatabaseSetup()
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QGroupBox {
                margin-top: 16px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
            QScrollArea {
                border: none;
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)
        
        # Main splitter for navigation and content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.layout.addWidget(main_splitter, 1)
        
        # Navigation
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_group = QGroupBox("Navigation")
        nav_group_layout = QVBoxLayout(nav_group)
        self.navigation_list = QListWidget()
        self.navigation_list.setIconSize(QSize(20,20))
        self.navigation_list.currentRowChanged.connect(self.change_page)
        nav_group_layout.addWidget(self.navigation_list)
        nav_layout.addWidget(nav_group)
        
        # Add save/load buttons to navigation area
        buttons_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.setIcon(load_bootstrap_icon("save"))
        self.save_button.clicked.connect(self.save_study_design)
        
        self.load_button = QPushButton("Load")
        self.load_button.setIcon(load_bootstrap_icon("folder2-open"))
        self.load_button.clicked.connect(self.load_study_design)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.setIcon(load_bootstrap_icon("arrow-counterclockwise"))
        self.reset_button.clicked.connect(self.reset_all_data)
        
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.reset_button)
        
        nav_layout.addLayout(buttons_layout)
        main_splitter.addWidget(nav_container)
        nav_container.setMaximumWidth(280)
        
        # Content area with scroll support
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        content_widget = QWidget()
        content_widget_layout = QVBoxLayout(content_widget)
        self.stacked_widget = QStackedWidget()
        content_widget_layout.addWidget(self.stacked_widget)
        scroll_area.setWidget(content_widget)
        content_layout.addWidget(scroll_area)
        main_splitter.addWidget(content_container)
        main_splitter.setSizes([25, 75])
        
        # Create tabs (only Eligibility section is modified below; others remain similar)
        self.timepoints_tab = QWidget()
        self.arms_tab = QWidget()
        self.outcomes_tab = QWidget()
        self.covariates_tab = QWidget()
        self.randomization_tab = QWidget()
        self.blinding_tab = QWidget()
        self.adverse_events_tab = QWidget()
        self.data_management_tab = QWidget()
        self.ethics_tab = QWidget()
        self.registration_tab = QWidget()
        self.data_collection_tab = QWidget()
        self.safety_monitoring_tab = QWidget()
        
        self.setup_timepoints_tab()
        self.setup_arms_tab()
        self.setup_outcomes_tab()
        self.setup_covariates_tab()
        self.setup_randomization_tab()
        self.setup_blinding_tab()
        self.setup_adverse_events_tab()
        self.setup_data_management_tab()
        self.setup_ethics_tab()
        self.setup_registration_tab()
        self.setup_safety_monitoring_tab()
        
        # Add navigation categories and items
        self.add_navigation_category("Study Design")
        self.add_navigation_item("Timepoints", self.timepoints_tab, "")
        self.add_navigation_item("Arms", self.arms_tab, "diagram-3")
        self.add_navigation_item("Outcome Measures", self.outcomes_tab, "activity")
        self.add_navigation_item("Covariates", self.covariates_tab, "sliders")
        self.add_navigation_item("Randomization & Blinding", self.randomization_tab, "shuffle")
        
        self.add_navigation_category("Safety")
        self.add_navigation_item("Adverse Events", self.adverse_events_tab, "exclamation-triangle")
        self.add_navigation_item("Safety Monitoring", self.safety_monitoring_tab, "heart")
        
        self.add_navigation_category("Administrative")
        self.add_navigation_item("Ethics", self.ethics_tab, "shield")
        self.add_navigation_item("Registration", self.registration_tab, "clipboard")
        
        self.set_placeholder_text_color()
        
    # --- Navigation Helpers ---
    def add_navigation_category(self, name: str):
        item = QListWidgetItem(name)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.navigation_list.addItem(item)
        
    def add_navigation_item(self, name: str, widget: QWidget, icon_name: str = None):
        item = QListWidgetItem("  " + name)
        if icon_name:
            item.setIcon(load_bootstrap_icon(icon_name))
        self.navigation_list.addItem(item)
        self.stacked_widget.addWidget(widget)
        
    def change_page(self, index):
        item = self.navigation_list.item(index)
        if item and item.flags() & Qt.ItemFlag.ItemIsSelectable:
            content_index = sum(1 for i in range(index) 
                               if self.navigation_list.item(i) and 
                               self.navigation_list.item(i).flags() & Qt.ItemFlag.ItemIsSelectable)
            self.stacked_widget.setCurrentIndex(content_index)
            
    def set_placeholder_text_color(self):
        stylesheet = """
            QLineEdit[placeholder="true"] {}
            QTextEdit[placeholder="true"] {}
        """
        self.setStyleSheet(self.styleSheet() + stylesheet)

    def populate_table_from_dataframe(self, table: QTableWidget, df: pd.DataFrame):
        """Helper method to populate a QTableWidget from a pandas DataFrame"""
        # Convert all values to strings with proper formatting
        def format_value(val):
            if pd.isna(val):
                return "NA"
            elif isinstance(val, (int, float)):
                return f"{val:.2f}" if isinstance(val, float) else str(val)
            return str(val)
        
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = format_value(df.iloc[i, j])
                item = QTableWidgetItem(value)
                table.setItem(i, j, item)
        
        # Add row numbers
        table.setVerticalHeaderLabels([str(i+1) for i in range(len(df))])
        
        # Adjust column widths to content
        table.resizeColumnsToContents()
        
        # Set alternating row colors for better readability
        table.setAlternatingRowColors(True)
        
        # Enable sorting
        table.setSortingEnabled(True)
        
        # Set minimum column widths
        for col in range(table.columnCount()):
            table.setColumnWidth(col, max(100, table.columnWidth(col)))

    def table_to_dataframe(self, table: QTableWidget) -> pd.DataFrame:
        """Convert QTableWidget to pandas DataFrame"""
        data = []
        headers = []
        
        # Get headers
        for j in range(table.columnCount()):
            headers.append(table.horizontalHeaderItem(j).text())
        
        # Get data
        for i in range(table.rowCount()):
            row = []
            for j in range(table.columnCount()):
                if table.item(i, j):
                    row.append(table.item(i, j).text())
                elif table.cellWidget(i, j):
                    # Handle comboboxes and other widgets
                    if isinstance(table.cellWidget(i, j), QComboBox):
                        row.append(table.cellWidget(i, j).currentText())
                    else:
                        row.append(str(table.cellWidget(i, j)))
                else:
                    row.append("")
            data.append(row)
            
        return pd.DataFrame(data, columns=headers)

    def configure_table_columns(self, table: QTableWidget, column_proportions=None):
        """Configure table to use full width with appropriately sized columns.
        
        Args:
            table: The QTableWidget to configure
            column_proportions: Optional list of relative width proportions for columns.
                                If None, columns will be distributed equally.
        """
        # Make table stretch to fill available space
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Set header properties for better readability
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionsClickable(True)
        
        # Set row height for better readability
        table.verticalHeader().setDefaultSectionSize(30)
        
        # Apply custom width proportions if provided
        if column_proportions:
            # Calculate total proportion
            total_proportion = sum(column_proportions)
            
            # Calculate the total available width
            table_width = table.viewport().width()
            
            # First set all columns to stretch mode
            for i in range(table.columnCount()):
                table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            
            # Then set fixed widths based on proportions for all except the last column
            for i in range(table.columnCount() - 1):
                if i < len(column_proportions):
                    width = int((column_proportions[i] / total_proportion) * table_width)
                    table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
                    table.setColumnWidth(i, width)
            
        # Make headers bold and centered
        header_font = table.horizontalHeader().font()
        header_font.setBold(True)
        table.horizontalHeader().setFont(header_font)
        
        # Set alternating row colors for better readability
        table.setAlternatingRowColors(True)
        
        # Enable sorting
        table.setSortingEnabled(True)
        
        # Connect to resize event to maintain proportions when window is resized
        table.horizontalHeader().sectionResized.connect(
            lambda index, oldSize, newSize: self.adjust_table_columns(table, column_proportions)
        )
    
    def adjust_table_columns(self, table, column_proportions):
        """Adjust column widths when table is resized to maintain proportions"""
        if not column_proportions:
            return
            
        # Only adjust if we're not in the middle of a resize operation
        if not hasattr(table, "_resizing") or not table._resizing:
            table._resizing = True
            
            # Calculate the total available width
            table_width = table.viewport().width()
            total_proportion = sum(column_proportions)
            
            # Set widths based on proportions for all except the last column
            for i in range(table.columnCount() - 1):
                if i < len(column_proportions):
                    width = int((column_proportions[i] / total_proportion) * table_width)
                    table.setColumnWidth(i, width)
            
            table._resizing = False

    def populate_config(self, config: dict):
        try:
            # Convert study type to string if it's an enum
            study_type = config.study_type
            if hasattr(study_type, 'value'):
                study_type = study_type.value
            
            # Find and select the study type in the combo box
            study_type_index = self.study_type_combo.findText(study_type)
            if study_type_index >= 0:
                self.study_type_combo.setCurrentIndex(study_type_index)

            # Populate timepoints
            self.timepoints_table.setRowCount(0)
            for timepoint in config.timepoints:
                row_position = self.timepoints_table.rowCount()
                self.timepoints_table.insertRow(row_position)
                
                self.timepoints_table.setItem(row_position, 0, QTableWidgetItem(timepoint.name))
                
                type_combo = QComboBox()
                type_combo.addItems([tp.value for tp in TimePoint])
                type_combo.setCurrentText(timepoint.point_type.value if hasattr(timepoint.point_type, 'value') else str(timepoint.point_type))
                self.timepoints_table.setCellWidget(row_position, 1, type_combo)
                
                self.timepoints_table.setItem(row_position, 2, QTableWidgetItem(str(timepoint.order)))
                self.timepoints_table.setItem(row_position, 3, QTableWidgetItem(str(timepoint.offset_days)))
                self.timepoints_table.setItem(row_position, 4, QTableWidgetItem(timepoint.description or ""))
                self.timepoints_table.setItem(row_position, 5, QTableWidgetItem(str(timepoint.window_days)))

            # Populate arms and interventions
            self.arms_table.setRowCount(0)
            for arm in config.arms:
                row_position = self.arms_table.rowCount()
                self.arms_table.insertRow(row_position)
                
                self.arms_table.setItem(row_position, 0, QTableWidgetItem(arm.name))
                
                interventions = '; '.join([
                    f"{i.name} ({i.type})" 
                    for i in arm.interventions
                ])
                self.arms_table.setItem(row_position, 1, QTableWidgetItem(interventions))
                
                self.arms_table.setItem(row_position, 2, QTableWidgetItem(arm.description))
                self.arms_table.setItem(row_position, 3, QTableWidgetItem(str(arm.start_date)))
                self.arms_table.setItem(row_position, 4, QTableWidgetItem(str(arm.end_date)))
                self.arms_table.setItem(row_position, 5, QTableWidgetItem(str(arm.cohort_size)))

            # Populate outcomes
            self.outcomes_table.setRowCount(0)
            for outcome in config.outcome_measures:
                row_position = self.outcomes_table.rowCount()
                self.outcomes_table.insertRow(row_position)
                
                self.outcomes_table.setItem(row_position, 0, QTableWidgetItem(outcome.name))
                self.outcomes_table.setItem(row_position, 1, QTableWidgetItem(outcome.description))
                self.outcomes_table.setItem(row_position, 2, QTableWidgetItem(', '.join(outcome.timepoints)))
                
                data_type_combo = QComboBox()
                data_type_combo.addItems([dt.value for dt in CFDataType])
                data_type_combo.setCurrentText(outcome.data_type.value)
                self.outcomes_table.setCellWidget(row_position, 3, data_type_combo)
                
                category_combo = QComboBox()
                category_combo.addItems([oc.value for oc in OutcomeCategory])
                category_combo.setCurrentText(outcome.category.value)
                self.outcomes_table.setCellWidget(row_position, 4, category_combo)
                
                collection_method_combo = QComboBox()
                collection_method_combo.addItems([cm.value for cm in DataCollectionMethod])
                collection_method_combo.setCurrentText(outcome.data_collection_method.value if outcome.data_collection_method and hasattr(outcome.data_collection_method, 'value') else "")
                self.outcomes_table.setCellWidget(row_position, 5, collection_method_combo)
                
                self.outcomes_table.setItem(row_position, 6, QTableWidgetItem(', '.join(outcome.applicable_arms)))
                self.outcomes_table.setItem(row_position, 7, QTableWidgetItem(str(outcome.units)))

            # Populate covariates
            self.covariates_table.setRowCount(0)
            for covariate in config.covariates:
                row_position = self.covariates_table.rowCount()
                self.covariates_table.insertRow(row_position)
                
                self.covariates_table.setItem(row_position, 0, QTableWidgetItem(covariate.name))
                self.covariates_table.setItem(row_position, 1, QTableWidgetItem(covariate.description))
                
                data_type_combo = QComboBox()
                data_type_combo.addItems([dt.value for dt in CFDataType])
                data_type_combo.setCurrentText(covariate.data_type.value if isinstance(covariate.data_type, CFDataType) else str(covariate.data_type))
                self.covariates_table.setCellWidget(row_position, 2, data_type_combo)

            # Populate randomization
            rand_scheme = config.randomization_scheme
            method = rand_scheme.method
            index = self.method_combo.findText(method)
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
                
            self.block_size_input.setText(str(rand_scheme.block_size))
            self.stratification_input.setText(', '.join(rand_scheme.stratification_factors))
            self.random_seed_input.setText(str(rand_scheme.random_seed))
            self.ratio_input.setText(', '.join(map(str, rand_scheme.randomization_ratio)))

            # Resize all tables and apply minimum width
            for table in [self.timepoints_table, self.arms_table, self.outcomes_table, 
                        self.covariates_table]:
                table.resizeColumnsToContents()
                # Set minimum column widths
                for col in range(table.columnCount()):
                    table.setColumnWidth(col, max(100, table.columnWidth(col)))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to populate study design: {str(e)}")

    def toggle_navigation(self):
        nav_container = self.navigation_list.parent().parent()
        nav_container.setVisible(not nav_container.isVisible())

    def create_form_layout(self, fields):
        """Helper to create a standardized form layout with proper spacing"""
        form_layout = QGridLayout()
        form_layout.setColumnStretch(1, 1)  # Make the second column (input fields) stretch
        form_layout.setColumnMinimumWidth(0, 200)  # Minimum width for labels
        form_layout.setSpacing(10)
        
        for row, (label_text, widget) in enumerate(fields):
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            form_layout.addWidget(label, row, 0)
            form_layout.addWidget(widget, row, 1)
            
        return form_layout

    def create_compact_form_layout(self, fields, columns=2):
        """Creates a more compact form layout with labels in their own row and multiple columns of widgets.
        
        Args:
            fields: List of tuples (label_text, widget)
            columns: Number of columns to arrange the widgets in
        """
        form_layout = QGridLayout()
        form_layout.setSpacing(10)
        
        current_row = 0
        for i in range(0, len(fields), columns):
            # Add labels in their own row
            for col in range(min(columns, len(fields) - i)):
                label_text, _ = fields[i + col]
                label = QLabel(label_text)
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
                form_layout.addWidget(label, current_row, col)
            
            current_row += 1
            
            # Add widgets in the next row
            for col in range(min(columns, len(fields) - i)):
                _, widget = fields[i + col]
                form_layout.addWidget(widget, current_row, col)
                form_layout.setColumnStretch(col, 1)  # Make all columns stretch equally
            
            current_row += 1
        
        return form_layout

    def setup_timepoints_tab(self):
        layout = QVBoxLayout(self.timepoints_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Study Timepoints")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Define the timepoints when assessments will be collected during the study.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        # Add inline timepoint form at the bottom of the tab
        self.timepoint_form_group = QGroupBox("Add New Timepoint")
        timepoint_form_layout = QVBoxLayout()
        self.timepoint_form_group.setLayout(timepoint_form_layout)
        
        form = QFormLayout()
        
        self.new_timepoint_name = QLineEdit()
        self.new_timepoint_description = QTextEdit()
        self.new_timepoint_description.setMaximumHeight(80)
        self.new_timepoint_time = QLineEdit()
        self.new_timepoint_unit = QComboBox()
        self.new_timepoint_unit.addItems(["Days", "Weeks", "Months", "Years"])
        
        form.addRow("Name:", self.new_timepoint_name)
        form.addRow("Description:", self.new_timepoint_description)
        form.addRow("Time:", self.new_timepoint_time)
        form.addRow("Unit:", self.new_timepoint_unit)
        
        timepoint_form_layout.addLayout(form)
        
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Timepoint")
        add_button.setIcon(load_bootstrap_icon("plus-circle"))
        add_button.clicked.connect(self.add_timepoint)
        clear_button = QPushButton("Clear")
        clear_button.setIcon(load_bootstrap_icon("x-circle"))
        clear_button.clicked.connect(self.clear_timepoint_form)
        remove_button = QPushButton("Remove Timepoint")
        remove_button.setIcon(load_bootstrap_icon("trash"))
        remove_button.clicked.connect(self.remove_timepoint)

        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)
        button_layout.addWidget(clear_button)
        timepoint_form_layout.addLayout(button_layout)
        
        layout.addWidget(self.timepoint_form_group)
        # Timepoints Table within a group box
        table_group = QGroupBox("Timepoints List")
        table_layout = QVBoxLayout(table_group)
        
        self.timepoints_table = QTableWidget()
        self.timepoints_table.setMinimumHeight(200)
        self.timepoints_table.setColumnCount(7)
        self.timepoints_table.setHorizontalHeaderLabels([
            "Name", "Type", "Order", "Offset Days", "Description", "Window Days", "Custom Name"
        ])
        
        # Configure table columns with proportional widths
        # Description gets more space (3), other columns get proportional space
        self.configure_table_columns(self.timepoints_table, [1.2, 1, 0.7, 1, 3, 1, 1.2])
        
        # Set alternating row colors for better readability (now handled by configure_table_columns)
        table_layout.addWidget(self.timepoints_table)
        
        layout.addWidget(table_group)
        layout.addStretch()

    def clear_timepoint_form(self):
        """Clear the inline timepoint form fields"""
        self.new_timepoint_name.clear()
        self.new_timepoint_description.clear()
        self.new_timepoint_time.clear()
        self.new_timepoint_unit.setCurrentIndex(0)

    def add_timepoint(self):
        """Add a timepoint using the inline form"""
        name = self.new_timepoint_name.text().strip()
        description = self.new_timepoint_description.toPlainText().strip()
        
        # Validate time input
        time_text = self.new_timepoint_time.text().strip()
        try:
            time_value = float(time_text) if time_text else 0
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Time must be a number")
            return
        
        unit = self.new_timepoint_unit.currentText()
        
        if not name:
            QMessageBox.warning(self, "Missing Information", "Timepoint name is required")
            return
        
        # Add to table
        row_position = self.timepoints_table.rowCount()
        self.timepoints_table.insertRow(row_position)
        
        self.timepoints_table.setItem(row_position, 0, QTableWidgetItem(name))
        self.timepoints_table.setItem(row_position, 1, QTableWidgetItem(description))
        self.timepoints_table.setItem(row_position, 2, QTableWidgetItem(str(time_value)))
        self.timepoints_table.setItem(row_position, 3, QTableWidgetItem(unit))
        
        # Clear the form
        self.clear_timepoint_form()
        
        # Adjust column widths
        self.adjust_table_columns(self.timepoints_table, [0.25, 0.45, 0.15, 0.15])


    def remove_timepoint(self):
        selected_row = self.timepoints_table.currentRow()
        if selected_row >= 0:
            self.timepoints_table.removeRow(selected_row)

    def setup_arms_tab(self):
        layout = QVBoxLayout(self.arms_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Study Arms")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Define the arms of the study and their associated interventions.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        arm_form_group = QGroupBox("Add New Arm")
        arm_form_layout = QVBoxLayout()
        arm_form_group.setLayout(arm_form_layout)
        
        form = QFormLayout()
        
        new_arm_name = QLineEdit()
        new_arm_description = QTextEdit()
        new_arm_description.setMaximumHeight(80)
        new_arm_size = QLineEdit()
        new_arm_type = QComboBox()
        new_arm_type.addItems(["Experimental", "Control", "Placebo", "Other"])
        
        form.addRow("Name:", new_arm_name)
        form.addRow("Description:", new_arm_description)
        form.addRow("Size:", new_arm_size)
        form.addRow("Type:", new_arm_type)
        
        arm_form_layout.addLayout(form)
        layout.addWidget(arm_form_group)

        # Arms Table within a group box
        table_group = QGroupBox("Arms List")
        table_layout = QVBoxLayout(table_group)
        
        self.arms_table = QTableWidget()
        self.arms_table.setMinimumHeight(200)
        self.arms_table.setColumnCount(6)
        self.arms_table.setHorizontalHeaderLabels([
            "Name", "Interventions", "Description", "Start Date", "End Date", "Cohort Size"
        ])
        
        # Configure table columns with proportional widths
        # Interventions and Description get more space, other columns get proportional space
        self.configure_table_columns(self.arms_table, [1, 2, 2, 1, 1, 1])
        
        table_layout.addWidget(self.arms_table)

        # Buttons in a horizontal layout with some styling
        button_layout = QHBoxLayout()
        self.add_arm_button = QPushButton("Add Arm")
        self.add_arm_button.setIcon(load_bootstrap_icon("plus-circle"))
        self.remove_arm_button = QPushButton("Remove Arm")
        self.remove_arm_button.setIcon(load_bootstrap_icon("trash"))
        self.add_intervention_button = QPushButton("Add Intervention")
        self.add_intervention_button.setIcon(load_bootstrap_icon("plus"))
        
        button_layout.addWidget(self.add_arm_button)
        button_layout.addWidget(self.remove_arm_button)
        button_layout.addWidget(self.add_intervention_button)
        button_layout.addStretch()
        table_layout.addLayout(button_layout)
        
        layout.addWidget(table_group)
        layout.addStretch()

        # Connect buttons
        self.add_arm_button.clicked.connect(self.add_arm)
        self.remove_arm_button.clicked.connect(self.remove_arm)
        self.add_intervention_button.clicked.connect(self.add_intervention)

    def add_arm(self):
        # Dialog for arm details
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Arm")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        fields = []
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Treatment Arm, Control Arm")
        fields.append(("Name:", name_input))
        
        description_input = QTextEdit()
        description_input.setPlaceholderText("Brief description of the arm")
        description_input.setMinimumHeight(100)
        fields.append(("Description:", description_input))
        
        start_date_input = QLineEdit()
        start_date_input.setPlaceholderText("YYYY-MM-DD")
        fields.append(("Start Date:", start_date_input))
        
        end_date_input = QLineEdit()
        end_date_input.setPlaceholderText("YYYY-MM-DD")
        fields.append(("End Date:", end_date_input))
        
        cohort_size_input = QLineEdit()
        cohort_size_input.setPlaceholderText("e.g., 50, 100")
        fields.append(("Cohort Size:", cohort_size_input))
        
        form_layout = self.create_form_layout(fields)
        layout.addLayout(form_layout)

        # Note about interventions
        note_label = QLabel("Note: Interventions can be added after creating the arm.")
        note_label.setStyleSheet("color: #777; font-style: italic;")
        layout.addWidget(note_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            row_position = self.arms_table.rowCount()
            self.arms_table.insertRow(row_position)
            # Set values, leave interventions blank initially
            self.arms_table.setItem(row_position, 0, QTableWidgetItem(name_input.text()))
            self.arms_table.setItem(row_position, 1, QTableWidgetItem(""))  # Blank for now
            self.arms_table.setItem(row_position, 2, QTableWidgetItem(description_input.toPlainText()))
            self.arms_table.setItem(row_position, 3, QTableWidgetItem(start_date_input.text()))
            self.arms_table.setItem(row_position, 4, QTableWidgetItem(end_date_input.text()))
            self.arms_table.setItem(row_position, 5, QTableWidgetItem(cohort_size_input.text()))
            
            # Maintain minimum column widths
            column_widths = [120, 200, 200, 100, 100, 100]
            for col, width in enumerate(column_widths):
                self.arms_table.setColumnWidth(col, max(width, self.arms_table.columnWidth(col)))

    def remove_arm(self):
        selected_row = self.arms_table.currentRow()
        if selected_row >= 0:
            self.arms_table.removeRow(selected_row)

    def add_intervention(self):
        # Dialog for intervention details
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Intervention")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Check if an arm is selected
        selected_row = self.arms_table.currentRow()
        if selected_row < 0:
            layout.addWidget(QLabel("Please select an arm to add the intervention to."))
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            dialog.exec()
            return
            
        # Display the selected arm name
        arm_name = self.arms_table.item(selected_row, 0).text()
        arm_label = QLabel(f"Adding intervention to arm: <b>{arm_name}</b>")
        layout.addWidget(arm_label)
        
        fields = []
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Drug A, Placebo")
        fields.append(("Name:", name_input))
        
        type_combo = QComboBox()
        type_combo.addItems([it.value for it in InterventionType])
        fields.append(("Type:", type_combo))
        
        description_input = QTextEdit()
        description_input.setPlaceholderText("Detailed description of the intervention")
        description_input.setMinimumHeight(100)
        fields.append(("Description:", description_input))
        
        form_layout = self.create_form_layout(fields)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get data from the dialog
            name = name_input.text()
            description = description_input.toPlainText()
            intervention_type = type_combo.currentText()

            # Add to the selected arm's interventions
            current_interventions = self.arms_table.item(selected_row, 1).text()
            new_interventions = f"{current_interventions}; {name} ({intervention_type}) - {description}" if current_interventions else f"{name} ({intervention_type}) - {description}"
            self.arms_table.item(selected_row, 1).setText(new_interventions)
            
            # Maintain minimum column widths
            column_widths = [120, 200, 200, 100, 100, 100]
            for col, width in enumerate(column_widths):
                self.arms_table.setColumnWidth(col, max(width, self.arms_table.columnWidth(col)))

    def setup_outcomes_tab(self):
        layout = QVBoxLayout(self.outcomes_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        title_label = QLabel("Outcome Measures")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Define the primary and secondary outcomes that will be measured in this study.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        # Outcomes Table within a group box
        table_group = QGroupBox("Outcomes List")
        table_layout = QVBoxLayout(table_group)
        
        self.outcomes_table = QTableWidget()
        self.outcomes_table.setMinimumHeight(200)
        self.outcomes_table.setColumnCount(8)
        self.outcomes_table.setHorizontalHeaderLabels([
            "Name", "Description", "Timepoints", "Data Type", "Category",
            "Collection Method", "Applicable Arms", "Units"
        ])
        
        # Configure table columns with proportional widths
        # Description gets more space, other columns get proportional space
        self.configure_table_columns(self.outcomes_table, [1.5, 2, 1.5, 1, 1, 1.5, 1.5, 1])
        
        table_layout.addWidget(self.outcomes_table)

        # Buttons in a horizontal layout with some styling
        button_layout = QHBoxLayout()
        self.add_outcome_button = QPushButton("Add Outcome")
        self.add_outcome_button.setIcon(load_bootstrap_icon("plus-circle"))
        self.remove_outcome_button = QPushButton("Remove Outcome")
        self.remove_outcome_button.setIcon(load_bootstrap_icon("trash"))
        
        button_layout.addWidget(self.add_outcome_button)
        button_layout.addWidget(self.remove_outcome_button)
        button_layout.addStretch()
        table_layout.addLayout(button_layout)
        
        layout.addWidget(table_group)
        layout.addStretch()

        # Connect buttons
        self.add_outcome_button.clicked.connect(self.add_outcome)
        self.remove_outcome_button.clicked.connect(self.remove_outcome)

    def add_outcome(self):
        # Dialog for outcome details
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Outcome Measure")
        dialog.setMinimumWidth(600)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        fields = []
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., FEV1, Quality of Life Score")
        fields.append(("Name:", name_input))
        
        description_input = QTextEdit()
        description_input.setPlaceholderText("Detailed description of the outcome")
        description_input.setMinimumHeight(80)
        fields.append(("Description:", description_input))
        
        timepoints_input = QLineEdit()
        timepoints_input.setPlaceholderText("e.g., Baseline, Week 12, End of Study")
        fields.append(("Timepoints:", timepoints_input))
        
        data_type_combo = QComboBox()
        data_type_combo.addItems([dt.value for dt in CFDataType])
        fields.append(("Data Type:", data_type_combo))
        
        category_combo = QComboBox()
        category_combo.addItems([oc.value for oc in OutcomeCategory])
        fields.append(("Category:", category_combo))
        
        collection_method_combo = QComboBox()
        collection_method_combo.addItems([cm.value for cm in DataCollectionMethod])
        fields.append(("Collection Method:", collection_method_combo))
        
        applicable_arms_input = QLineEdit()
        applicable_arms_input.setPlaceholderText("e.g., All, Treatment Arm")
        fields.append(("Applicable Arms:", applicable_arms_input))
        
        units_input = QLineEdit()
        units_input.setPlaceholderText("e.g., L, %, score")
        fields.append(("Units:", units_input))
        
        # Create a grid layout with proper spacing and alignment
        form_layout = self.create_form_layout(fields)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            row_position = self.outcomes_table.rowCount()
            self.outcomes_table.insertRow(row_position)

            # Set values from dialog
            self.outcomes_table.setItem(row_position, 0, QTableWidgetItem(name_input.text()))
            self.outcomes_table.setItem(row_position, 1, QTableWidgetItem(description_input.toPlainText()))
            self.outcomes_table.setItem(row_position, 2, QTableWidgetItem(timepoints_input.text()))

            # Comboboxes
            self.outcomes_table.setCellWidget(row_position, 3, QComboBox())
            self.outcomes_table.cellWidget(row_position, 3).addItems([dt.value for dt in CFDataType])
            self.outcomes_table.cellWidget(row_position, 3).setCurrentText(data_type_combo.currentText())

            self.outcomes_table.setCellWidget(row_position, 4, QComboBox())
            self.outcomes_table.cellWidget(row_position, 4).addItems([oc.value for oc in OutcomeCategory])
            self.outcomes_table.cellWidget(row_position, 4).setCurrentText(category_combo.currentText())

            self.outcomes_table.setCellWidget(row_position, 5, QComboBox())
            self.outcomes_table.cellWidget(row_position, 5).addItems([cm.value for cm in DataCollectionMethod])
            self.outcomes_table.cellWidget(row_position, 5).setCurrentText(collection_method_combo.currentText())

            self.outcomes_table.setItem(row_position, 6, QTableWidgetItem(applicable_arms_input.text()))
            self.outcomes_table.setItem(row_position, 7, QTableWidgetItem(units_input.text()))
            
            # Maintain minimum column widths
            column_widths = [150, 200, 150, 120, 120, 150, 150, 100]
            for col, width in enumerate(column_widths):
                self.outcomes_table.setColumnWidth(col, max(width, self.outcomes_table.columnWidth(col)))

    def remove_outcome(self):
        selected_row = self.outcomes_table.currentRow()
        if selected_row >= 0:
            self.outcomes_table.removeRow(selected_row)

    def setup_covariates_tab(self):
        layout = QVBoxLayout(self.covariates_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Covariates")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Define the covariates that will be measured or controlled for in the study.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        # Covariates Table within a group box
        table_group = QGroupBox("Covariates List")
        table_layout = QVBoxLayout(table_group)
        
        self.covariates_table = QTableWidget()
        self.covariates_table.setMinimumHeight(200)
        self.covariates_table.setColumnCount(3)
        self.covariates_table.setHorizontalHeaderLabels(["Name", "Description", "Data Type"])
        
        # Configure table columns with proportional widths
        # Description gets more space, other columns get proportional space
        self.configure_table_columns(self.covariates_table, [1.5, 3.5, 1.5])
        
        table_layout.addWidget(self.covariates_table)

        # Buttons in a horizontal layout with some styling
        button_layout = QHBoxLayout()
        self.add_covariate_button = QPushButton("Add Covariate")
        self.add_covariate_button.setIcon(load_bootstrap_icon("plus-circle"))
        self.remove_covariate_button = QPushButton("Remove Covariate")
        self.remove_covariate_button.setIcon(load_bootstrap_icon("trash"))
        
        button_layout.addWidget(self.add_covariate_button)
        button_layout.addWidget(self.remove_covariate_button)
        button_layout.addStretch()
        table_layout.addLayout(button_layout)
        
        layout.addWidget(table_group)
        layout.addStretch()

        # Connect buttons
        self.add_covariate_button.clicked.connect(self.add_covariate)
        self.remove_covariate_button.clicked.connect(self.remove_covariate)

    def add_covariate(self):
        # Dialog for covariate details
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Covariate")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        fields = []
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Age, Gender, BMI")
        fields.append(("Name:", name_input))
        
        description_input = QTextEdit()
        description_input.setPlaceholderText("Detailed description of the covariate")
        description_input.setMinimumHeight(100)
        fields.append(("Description:", description_input))
        
        data_type_combo = QComboBox()
        data_type_combo.addItems([dt.value for dt in CFDataType])
        fields.append(("Data Type:", data_type_combo))
        
        # Create a grid layout with proper spacing and alignment
        form_layout = self.create_form_layout(fields)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            row_position = self.covariates_table.rowCount()
            self.covariates_table.insertRow(row_position)

            # Set values from dialog
            self.covariates_table.setItem(row_position, 0, QTableWidgetItem(name_input.text()))
            self.covariates_table.setItem(row_position, 1, QTableWidgetItem(description_input.toPlainText()))

            # Combobox
            self.covariates_table.setCellWidget(row_position, 2, QComboBox())
            self.covariates_table.cellWidget(row_position, 2).addItems([dt.value for dt in CFDataType])
            self.covariates_table.cellWidget(row_position, 2).setCurrentText(data_type_combo.currentText())
            
            # Maintain minimum column widths
            column_widths = [150, 350, 150]
            for col, width in enumerate(column_widths):
                self.covariates_table.setColumnWidth(col, max(width, self.covariates_table.columnWidth(col)))

    def remove_covariate(self):
        selected_row = self.covariates_table.currentRow()
        if selected_row >= 0:
            self.covariates_table.removeRow(selected_row)

    def setup_randomization_tab(self):
        layout = QVBoxLayout(self.randomization_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Randomization and Blinding")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Define how participants will be randomized to study arms and the level of blinding implemented.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        # Main content with two group boxes side by side
        main_content = QHBoxLayout()
        layout.addLayout(main_content)
        
        # Left side: Randomization settings
        settings_group = QGroupBox("Randomization Configuration")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)
        
        # Create compact form fields for randomization
        rand_fields = []
        
        # Method
        self.method_combo = QComboBox()
        self.method_combo.addItems([rm.value for rm in RandomizationMethod])
        rand_fields.append(("Method", self.method_combo))
        
        # Block Size
        self.block_size_input = QLineEdit()
        self.block_size_input.setPlaceholderText("e.g., 4, 6, 8")
        rand_fields.append(("Block Size", self.block_size_input))
        
        # Stratification Factors
        self.stratification_input = QLineEdit()
        self.stratification_input.setPlaceholderText("e.g., Age, Gender, Center")
        rand_fields.append(("Stratification Factors", self.stratification_input))
        
        # Random Seed
        self.random_seed_input = QLineEdit()
        self.random_seed_input.setPlaceholderText("e.g., 42, 12345")
        rand_fields.append(("Random Seed", self.random_seed_input))
        
        # Randomization Ratio
        self.ratio_input = QLineEdit()
        self.ratio_input.setPlaceholderText("e.g., 1, 1 for 1:1 ratio")
        rand_fields.append(("Randomization Ratio", self.ratio_input))
        
        # Create a compact grid layout with 2 columns
        rand_form_layout = self.create_compact_form_layout(rand_fields, columns=2)
        settings_layout.addLayout(rand_form_layout)
        
        # Add description text area with its own label in a row
        desc_label = QLabel("Description")
        settings_layout.addWidget(desc_label)
        
        self.randomization_description_input = QTextEdit()
        self.randomization_description_input.setPlaceholderText("Additional details about randomization procedures")
        self.randomization_description_input.setMinimumHeight(100)
        settings_layout.addWidget(self.randomization_description_input)
        
        main_content.addWidget(settings_group, 2)  # Give it more space
        
        # Right side: Blinding settings
        blinding_group = QGroupBox("Blinding Configuration")
        blinding_layout = QVBoxLayout(blinding_group)
        blinding_layout.setSpacing(12)
        
        # Blinding Type
        blinding_label = QLabel("Blinding Type")
        blinding_layout.addWidget(blinding_label)
        
        self.blinding_combo = QComboBox()
        self.blinding_combo.addItems([bt.value for bt in BlindingType])
        blinding_layout.addWidget(self.blinding_combo)
        
        # Add description text area
        desc_label = QLabel("Blinding Details")
        blinding_layout.addWidget(desc_label)
        
        self.blinding_description_input = QTextEdit()
        self.blinding_description_input.setPlaceholderText("Describe which parties are blinded and how blinding is maintained")
        self.blinding_description_input.setMinimumHeight(100)
        blinding_layout.addWidget(self.blinding_description_input)
        
        main_content.addWidget(blinding_group, 1)  # Give it less space
        
        # Add a save button
        save_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        save_button.setIcon(load_bootstrap_icon("save"))
        save_layout.addStretch()
        save_layout.addWidget(save_button)
        layout.addLayout(save_layout)
        
        layout.addStretch()
        
        # Connect save button
        save_button.clicked.connect(self.save_randomization_settings)

    def save_randomization_settings(self):
        """Save the randomization settings"""
        # Here you would implement the logic to save the randomization settings
        QMessageBox.information(self, "Settings Saved", "Randomization settings have been saved successfully.")

    def setup_blinding_tab(self):
        layout = QVBoxLayout(self.blinding_tab)

        # Blinding Type
        blinding_layout = QHBoxLayout()
        self.blinding_label = QLabel("Blinding Type:")
        self.blinding_combo = QComboBox()
        self.blinding_combo.addItems([bt.value for bt in BlindingType])
        blinding_layout.addWidget(self.blinding_label)
        blinding_layout.addWidget(self.blinding_combo)
        layout.addLayout(blinding_layout)
        layout.addStretch()

    def setup_adverse_events_tab(self):
        layout = QVBoxLayout(self.adverse_events_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Adverse Events")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Record and track adverse events that occur during the study.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        # Adverse Events Table within a group box
        table_group = QGroupBox("Adverse Events List")
        table_layout = QVBoxLayout(table_group)

        # Adverse Events Table
        self.adverse_events_table = QTableWidget()
        self.adverse_events_table.setMinimumHeight(200)
        self.adverse_events_table.setColumnCount(8)
        self.adverse_events_table.setHorizontalHeaderLabels([
            "Participant ID", "Description", "Severity", "Causality",
            "Intervention", "Onset Date", "Resolution Date", "Action Taken"
        ])
        
        # Configure table columns with proportional widths
        # Description gets more space, dates get less
        self.configure_table_columns(self.adverse_events_table, [1, 2.5, 1, 1, 1.5, 1, 1, 1.5])
        
        table_layout.addWidget(self.adverse_events_table)

        # Buttons in a horizontal layout with some styling
        button_layout = QHBoxLayout()
        self.add_adverse_event_button = QPushButton("Add Adverse Event")
        self.add_adverse_event_button.setIcon(load_bootstrap_icon("plus-circle"))
        self.remove_adverse_event_button = QPushButton("Remove Adverse Event")
        self.remove_adverse_event_button.setIcon(load_bootstrap_icon("trash"))
        
        button_layout.addWidget(self.add_adverse_event_button)
        button_layout.addWidget(self.remove_adverse_event_button)
        button_layout.addStretch()
        table_layout.addLayout(button_layout)
        
        layout.addWidget(table_group)
        layout.addStretch()

        # Connect buttons
        self.add_adverse_event_button.clicked.connect(self.add_adverse_event)
        self.remove_adverse_event_button.clicked.connect(self.remove_adverse_event)

    def add_adverse_event(self):
        # Dialog for adverse event
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Adverse Event")
        layout = QVBoxLayout(dialog)

        participant_id_layout = QHBoxLayout()
        participant_id_label = QLabel("Participant ID:")
        participant_id_input = QLineEdit()
        participant_id_input.setPlaceholderText("Unique ID of the participant")
        participant_id_layout.addWidget(participant_id_label)
        participant_id_layout.addWidget(participant_id_input)
        layout.addLayout(participant_id_layout)

        description_layout = QHBoxLayout()
        description_label = QLabel("Description:")
        description_input = QTextEdit()
        description_input.setPlaceholderText("Detailed description of the adverse event")
        description_layout.addWidget(description_label)
        description_layout.addWidget(description_input)
        layout.addLayout(description_layout)

        severity_layout = QHBoxLayout()
        severity_label = QLabel("Severity:")
        severity_combo = QComboBox()
        severity_combo.addItems([sev.value for sev in AdverseEventSeverity])
        severity_layout.addWidget(severity_label)
        severity_layout.addWidget(severity_combo)
        layout.addLayout(severity_layout)

        causality_layout = QHBoxLayout()
        causality_label = QLabel("Causality:")
        causality_combo = QComboBox()
        causality_combo.addItems([caus.value for caus in AdverseEventCausality])
        causality_layout.addWidget(causality_label)
        causality_layout.addWidget(causality_combo)
        layout.addLayout(causality_layout)

        intervention_layout = QHBoxLayout()
        intervention_label = QLabel("Intervention:")
        intervention_input = QLineEdit()
        intervention_input.setPlaceholderText("Related intervention if applicable")
        intervention_layout.addWidget(intervention_label)
        intervention_layout.addWidget(intervention_input)
        layout.addLayout(intervention_layout)

        onset_date_layout = QHBoxLayout()
        onset_date_label = QLabel("Onset Date:")
        onset_date_input = QLineEdit()
        onset_date_input.setPlaceholderText("YYYY-MM-DD")
        onset_date_layout.addWidget(onset_date_label)
        onset_date_layout.addWidget(onset_date_input)
        layout.addLayout(onset_date_layout)

        resolution_date_layout = QHBoxLayout()
        resolution_date_label = QLabel("Resolution Date:")
        resolution_date_input = QLineEdit()
        resolution_date_input.setPlaceholderText("YYYY-MM-DD (leave blank if ongoing)")
        resolution_date_layout.addWidget(resolution_date_label)
        resolution_date_layout.addWidget(resolution_date_input)
        layout.addLayout(resolution_date_layout)

        action_taken_layout = QHBoxLayout()
        action_taken_label = QLabel("Action Taken:")
        action_taken_input = QLineEdit()
        action_taken_input.setPlaceholderText("Actions taken in response to the event")
        action_taken_layout.addWidget(action_taken_label)
        action_taken_layout.addWidget(action_taken_input)
        layout.addLayout(action_taken_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            row_position = self.adverse_events_table.rowCount()
            self.adverse_events_table.insertRow(row_position)

            # Set values
            self.adverse_events_table.setItem(row_position, 0, QTableWidgetItem(participant_id_input.text()))
            self.adverse_events_table.setItem(row_position, 1, QTableWidgetItem(description_input.toPlainText()))

            self.adverse_events_table.setCellWidget(row_position, 2, QComboBox())
            self.adverse_events_table.cellWidget(row_position, 2).addItems([sev.value for sev in AdverseEventSeverity])
            self.adverse_events_table.cellWidget(row_position, 2).setCurrentText(severity_combo.currentText())

            self.adverse_events_table.setCellWidget(row_position, 3, QComboBox())
            self.adverse_events_table.cellWidget(row_position, 3).addItems([caus.value for caus in AdverseEventCausality])
            self.adverse_events_table.cellWidget(row_position, 3).setCurrentText(causality_combo.currentText())

            self.adverse_events_table.setItem(row_position, 4, QTableWidgetItem(intervention_input.text()))
            self.adverse_events_table.setItem(row_position, 5, QTableWidgetItem(onset_date_input.text()))
            self.adverse_events_table.setItem(row_position, 6, QTableWidgetItem(resolution_date_input.text()))
            self.adverse_events_table.setItem(row_position, 7, QTableWidgetItem(action_taken_input.text()))
            
            # Reconfigure table columns to maintain proportions
            self.configure_table_columns(self.adverse_events_table, [1, 2.5, 1, 1, 1.5, 1, 1, 1.5])

    def remove_adverse_event(self):
        selected_row = self.adverse_events_table.currentRow()
        if selected_row >= 0:
            self.adverse_events_table.removeRow(selected_row)

    def setup_data_management_tab(self):
        layout = QVBoxLayout(self.data_management_tab)

        # Data Collection Tools
        tools_layout = QHBoxLayout()
        self.data_tools_label = QLabel("Data Collection Tools (comma-separated):")
        self.data_tools_input = QLineEdit()
        tools_layout.addWidget(self.data_tools_label)
        tools_layout.addWidget(self.data_tools_input)
        layout.addLayout(tools_layout)

        # Data Storage Location
        storage_layout = QHBoxLayout()
        self.storage_label = QLabel("Data Storage Location:")
        self.storage_input = QLineEdit()
        storage_layout.addWidget(self.storage_label)
        storage_layout.addWidget(self.storage_input)
        layout.addLayout(storage_layout)

        # Data Security Measures
        security_layout = QHBoxLayout()
        self.security_label = QLabel("Data Security Measures:")
        self.security_input = QTextEdit()
        security_layout.addWidget(self.security_label)
        security_layout.addWidget(self.security_input)
        layout.addLayout(security_layout)

        # Data Sharing Policy
        sharing_layout = QHBoxLayout()
        self.sharing_label = QLabel("Data Sharing Policy:")
        self.sharing_input = QTextEdit()
        sharing_layout.addWidget(self.sharing_label)
        sharing_layout.addWidget(self.sharing_input)
        layout.addLayout(sharing_layout)

        # Quality Control Measures
        qc_layout = QHBoxLayout()
        self.qc_label = QLabel("Quality Control Measures:")
        self.qc_input = QTextEdit()
        qc_layout.addWidget(self.qc_label)
        qc_layout.addWidget(self.qc_input)
        layout.addLayout(qc_layout)

        # Backup Procedures
        backup_layout = QHBoxLayout()
        self.backup_label = QLabel("Backup Procedures:")
        self.backup_input = QTextEdit()
        backup_layout.addWidget(self.backup_label)
        backup_layout.addWidget(self.backup_input)
        layout.addLayout(backup_layout)

        layout.addStretch()

    def setup_ethics_tab(self):
        layout = QVBoxLayout(self.ethics_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Ethics Approval")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Document the ethical review and approval for the study.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        # Ethics settings in a group box
        settings_group = QGroupBox("Ethics Committee Details")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)
        
        # Create fields for the first row
        id_fields = []
        
        # Committee Name
        self.committee_input = QLineEdit()
        self.committee_input.setPlaceholderText("e.g., University IRB")
        id_fields.append(("Committee Name", self.committee_input))
        
        # Approval ID
        self.approval_id_input = QLineEdit()
        self.approval_id_input.setPlaceholderText("e.g., IRB-12345")
        id_fields.append(("Approval ID", self.approval_id_input))
        
        # Use compact layout for first row
        id_form_layout = self.create_compact_form_layout(id_fields, columns=2)
        settings_layout.addLayout(id_form_layout)
        
        # Create fields for the date row
        date_fields = []
        
        # Approval Date
        self.approval_date_input = QLineEdit()
        self.approval_date_input.setPlaceholderText("YYYY-MM-DD")
        date_fields.append(("Approval Date", self.approval_date_input))
        
        # Expiration Date
        self.expiration_date_input = QLineEdit()
        self.expiration_date_input.setPlaceholderText("YYYY-MM-DD")
        date_fields.append(("Expiration Date", self.expiration_date_input))
        
        # Use compact layout for date row
        date_form_layout = self.create_compact_form_layout(date_fields, columns=2)
        settings_layout.addLayout(date_form_layout)
        
        # Add notes text area with its own label in a row
        notes_label = QLabel("Notes")
        settings_layout.addWidget(notes_label)
        
        self.ethics_notes_input = QTextEdit()
        self.ethics_notes_input.setPlaceholderText("Additional details about ethics approval or special considerations")
        self.ethics_notes_input.setMinimumHeight(100)
        settings_layout.addWidget(self.ethics_notes_input)
        
        layout.addWidget(settings_group)
        layout.addStretch()

    def setup_registration_tab(self):
        layout = QVBoxLayout(self.registration_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Add debugging print statement
        print("Setting up registration tab")
        
        # Add a title and description
        title_label = QLabel("Registration")
        title_label.setProperty("class", "section-title")
        description_label = QLabel("Document the registration details of your clinical trial.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        
        # Registration settings in a group box
        settings_group = QGroupBox("Registration Details")
        settings_layout = QVBoxLayout(settings_group)
        
        # Make sure the layout is properly set
        settings_group.setLayout(settings_layout)
        
        # Create form fields - ensure they're being added to the layout
        form_layout = QFormLayout()
        
        self.registry_input = QLineEdit()
        self.registry_input.setPlaceholderText("e.g., ClinicalTrials.gov")
        
        self.registration_id_input = QLineEdit()
        self.registration_id_input.setPlaceholderText("e.g., NCT01234567")
        
        self.registration_date_input = QDateEdit()
        self.registration_date_input.setCalendarPopup(True)
        self.registration_date_input.setDate(QDate.currentDate())
        
        # Add widgets to form layout
        form_layout.addRow("Registry:", self.registry_input)
        form_layout.addRow("Registration ID:", self.registration_id_input)
        form_layout.addRow("Registration Date:", self.registration_date_input)
        
        # Add form layout to settings layout
        settings_layout.addLayout(form_layout)
        
        # Add settings group to main layout
        layout.addWidget(settings_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()

    def save_study_design(self):
        """Save the entire study design to the database."""
        try:
            study_data = self.collect_study_data()
            study_id = self.db_setup.save_study_design(
                client_id=self.client_id,
                study_design_data=study_data
            )
            
            if study_id:
                self.db_setup.log_user_action(
                    client_id=self.client_id,
                    action_type="save_study_design",
                    details={"study_id": study_id}
                )
                QMessageBox.information(self, "Success", f"Study design saved successfully with ID: {study_id}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save study design")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving study design: {str(e)}")

    def collect_study_data(self):
        """Collect all study design data."""
        return {
            "timepoints": self.collect_timepoints_data(),
            "arms": self.collect_arms_data(),
            "outcomes": self.collect_outcomes_data(),
            "covariates": self.collect_covariates_data(),
            "randomization": self.collect_randomization_data(),
            "blinding": self.collect_blinding_data(),
            "adverse_events": self.collect_adverse_events_data(),
            "data_management": self.collect_data_management_data(),
            "ethics": self.collect_ethics_data(),
            "registration": self.collect_registration_data(),
            "safety_monitoring": self.collect_safety_monitoring_data(),
        }

    def get_text_safely(self, widget):
        """Safely get text from either QLineEdit or QTextEdit widget."""
        if hasattr(widget, 'toPlainText'):
            return widget.toPlainText()
        elif hasattr(widget, 'text'):
            return widget.text()
        return ''

    def collect_timepoints_data(self):
        """Collect data from the timepoints table."""
        timepoints = []
        for row in range(self.timepoints_table.rowCount()):
            timepoint = {
                "name": self.timepoints_table.item(row, 0).text(),
                "type": self.timepoints_table.cellWidget(row, 1).currentText(),
                "order": self.timepoints_table.item(row, 2).text(),
                "offset_days": self.timepoints_table.item(row, 3).text(),
                "description": self.timepoints_table.item(row, 4).text(),
                "window_days": self.timepoints_table.item(row, 5).text(),
            }
            timepoints.append(timepoint)
        return timepoints

    def collect_arms_data(self):
        """Collect data from the arms table."""
        arms = []
        for row in range(self.arms_table.rowCount()):
            arm = {
                "name": self.arms_table.item(row, 0).text(),
                "interventions": self.arms_table.item(row, 1).text().split(';') if self.arms_table.item(row, 1) else [],
                "description": self.arms_table.item(row, 2).text(),
                "start_date": self.arms_table.item(row, 3).text(),
                "end_date": self.arms_table.item(row, 4).text(),
                "cohort_size": self.arms_table.item(row, 5).text()
            }
            arms.append(arm)
        return arms

    def collect_outcomes_data(self):
        """Collect data from the outcomes table."""
        outcomes = []
        for row in range(self.outcomes_table.rowCount()):
            outcome = {
                "name": self.outcomes_table.item(row, 0).text(),
                "description": self.outcomes_table.item(row, 1).text(),
                "timepoints": self.outcomes_table.item(row, 2).text().split(',') if self.outcomes_table.item(row, 2) else [],
                "data_type": self.outcomes_table.item(row, 3).text(),
                "category": self.outcomes_table.item(row, 4).text(),
                "collection_method": self.outcomes_table.item(row, 5).text(),
                "applicable_arms": self.outcomes_table.item(row, 6).text().split(',') if self.outcomes_table.item(row, 6) else [],
                "units": self.outcomes_table.item(row, 7).text(),
            }
            outcomes.append(outcome)
        return outcomes

    def collect_covariates_data(self):
        """Collect data from the covariates table."""
        covariates = []
        for row in range(self.covariates_table.rowCount()):
            covariate = {
                "name": self.covariates_table.item(row, 0).text(),
                "description": self.covariates_table.item(row, 1).text(),
                "data_type": self.covariates_table.item(row, 2).text()
            }
            covariates.append(covariate)
        return covariates

    def collect_randomization_data(self):
        """Collect randomization data."""
        return {
            "method": self.method_combo.currentText(),
            "block_size": self.block_size_input.text() if hasattr(self, 'block_size_input') else None,
            "stratification_factors": self.stratification_input.text().split(',') if hasattr(self, 'stratification_input') else None,
            "randomization_description": self.randomization_description_input.toPlainText() if hasattr(self, 'randomization_description_input') else None,
            "random_seed": self.random_seed_input.text() if hasattr(self, 'random_seed_input') else None,
            "ratio": self.ratio_input.text() if hasattr(self, 'ratio_input') else None
        }

    def collect_blinding_data(self):
        """Collect blinding data."""
        return {
            "type": self.blinding_combo.currentText()
        }

    def collect_adverse_events_data(self):
        """Collect adverse events data."""
        events = []
        for row in range(self.adverse_events_table.rowCount()):
            event = {
                "participant_id": self.adverse_events_table.item(row, 0).text(),
                "description": self.adverse_events_table.item(row, 1).text(),
                "severity": self.adverse_events_table.cellWidget(row, 2).currentText(),
                "causality": self.adverse_events_table.cellWidget(row, 3).currentText(),
                "intervention": self.adverse_events_table.item(row, 4).text(),
                "onset_date": self.adverse_events_table.item(row, 5).text(),
                "resolution_date": self.adverse_events_table.item(row, 6).text(),
                "action_taken": self.adverse_events_table.item(row, 7).text()
            }
            events.append(event)
        return events

    def collect_data_management_data(self):
        """Collect data management plan."""
        return {
            "tools": self.data_tools_input.text().split(',') if hasattr(self, 'data_tools_input') else [],
            "storage_location": self.storage_input.text() if hasattr(self, 'storage_input') else None,
            "security_measures": self.security_input.toPlainText() if hasattr(self, 'security_input') else None,
            "sharing_policy": self.sharing_input.toPlainText() if hasattr(self, 'sharing_input') else None,
            "quality_control": self.qc_input.toPlainText() if hasattr(self, 'qc_input') else None,
            "backup_procedures": self.backup_input.toPlainText() if hasattr(self, 'backup_input') else None
        }

    def collect_ethics_data(self):
        """Collect ethics approval data."""
        return {
            "committee_name": self.committee_input.text() if hasattr(self, 'committee_input') else None,
            "approval_id": self.approval_id_input.text() if hasattr(self, 'approval_id_input') else None,
            "approval_date": self.approval_date_input.text() if hasattr(self, 'approval_date_input') else None,
            "expiration_date": self.expiration_date_input.text() if hasattr(self, 'expiration_date_input') else None,
            "notes": self.ethics_notes_input.toPlainText() if hasattr(self, 'ethics_notes_input') else None
        }

    def collect_registration_data(self):
        """Collect study registration data."""
        return {
            "registration_id": self.registration_id_input.text() if hasattr(self, 'registration_id_input') else None,
            "registration_date": self.registration_date_input.text() if hasattr(self, 'registration_date_input') else None,
        }

    def collect_safety_monitoring_data(self):
        """Collect safety monitoring data."""
        return {
            "dsmb_members": self.dsmb_input.text().split(',') if hasattr(self, 'dsmb_input') else [],
            "meeting_schedule": self.meeting_input.text() if hasattr(self, 'meeting_input') else None,
            "reporting_procedures": self.reporting_input.toPlainText() if hasattr(self, 'reporting_input') else None
        }

    def populate_arms_data(self, arms_data):
        """Populate the arms table with loaded data."""
        self.arms_table.setRowCount(0)
        for arm in arms_data:
            row_position = self.arms_table.rowCount()
            self.arms_table.insertRow(row_position)
            self.arms_table.setItem(row_position, 0, QTableWidgetItem(arm.get('name', '')))
            
            # Use the interventions_display field if available, otherwise try to format interventions
            if 'interventions_display' in arm:
                interventions_text = arm.get('interventions_display', '')
            else:
                interventions = arm.get('interventions', [])
                if isinstance(interventions, list):
                    # Try to format the interventions if they're objects or dictionaries
                    interventions_list = []
                    for intervention in interventions:
                        if isinstance(intervention, dict):
                            name = intervention.get('name', '')
                            int_type = intervention.get('type', '')
                            interventions_list.append(f"{name} ({int_type})")
                        elif isinstance(intervention, str):
                            interventions_list.append(intervention)
                        else:
                            interventions_list.append(str(intervention))
                    interventions_text = '; '.join(interventions_list)
                else:
                    interventions_text = str(interventions)
            
            self.arms_table.setItem(row_position, 1, QTableWidgetItem(interventions_text))
            self.arms_table.setItem(row_position, 2, QTableWidgetItem(arm.get('description', '')))
            self.arms_table.setItem(row_position, 3, QTableWidgetItem(str(arm.get('start_date', ''))))
            self.arms_table.setItem(row_position, 4, QTableWidgetItem(str(arm.get('end_date', ''))))
            self.arms_table.setItem(row_position, 5, QTableWidgetItem(str(arm.get('cohort_size', ''))))
        self.arms_table.resizeColumnsToContents()

    def populate_study_data(self, data):
        """Populate all UI elements with loaded study data."""
        # Populate general data (already implemented)
        
        # Populate other sections
        self.populate_timepoints_data(data.get('timepoints', []))
        self.populate_arms_data(data.get('arms', []))
        self.populate_outcomes_data(data.get('outcomes', []))
        self.populate_covariates_data(data.get('covariates', []))
        self.populate_randomization_data(data.get('randomization', {}))
        self.populate_blinding_data(data.get('blinding', {}))
        self.populate_adverse_events_data(data.get('adverse_events', []))
        self.populate_data_management_data(data.get('data_management', {}))
        self.populate_ethics_data(data.get('ethics', {}))
        self.populate_registration_data(data.get('registration', {}))
        self.populate_data_collection_data(data.get('data_collection', {}))
        self.populate_safety_monitoring_data(data.get('safety_monitoring', {}))

    def load_study_design(self):
        """Load a study design from the database with a dropdown selection."""
        try:
            # Get list of studies for this client
            study_designs_db = self.db_setup.server['study_designs']
            results = study_designs_db.view('study_designs/by_client_id', key=self.client_id)
            
            if not results:
                QMessageBox.information(self, "No Designs", "No saved study designs found for this user.")
                return

            # Create a dialog for study selection
            dialog = QDialog(self)
            dialog.setWindowTitle("Load Study Design")
            dialog.setMinimumWidth(400)
            layout = QVBoxLayout(dialog)

            # Add informative label
            label = QLabel("Select a study design to load:")
            layout.addWidget(label)

            # Create table widget for better display
            table = QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Study ID", "Title", "Last Modified", "Description"])
            table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(table)

            # Populate table with study designs
            study_list = []
            table.setRowCount(len(list(results)))
            for i, row in enumerate(results):
                doc = row.value
                study_data = doc['data']
                general_data = study_data.get('general', {})
                
                # Store full document for later use
                study_list.append(doc)
                
                # Add items to table
                table.setItem(i, 0, QTableWidgetItem(doc['_id']))
                table.setItem(i, 1, QTableWidgetItem(general_data.get('title', 'Untitled')))
                table.setItem(i, 2, QTableWidgetItem(doc.get('timestamp', 'Unknown')))
                table.setItem(i, 3, QTableWidgetItem(general_data.get('description', '')))

            # Adjust table properties
            table.resizeColumnsToContents()
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            
            # Add preview section
            preview_group = QGroupBox("Study Preview")
            preview_layout = QVBoxLayout(preview_group)
            preview_text = QTextEdit()
            preview_text.setReadOnly(True)
            preview_layout.addWidget(preview_text)
            layout.addWidget(preview_group)

            # After successful load and populate_study_data call:
            if selected_doc:
                self.populate_study_data(selected_doc['data'])
            # Update preview when selection changes
            def update_preview():
                selected_rows = table.selectedItems()
                if selected_rows:
                    row = selected_rows[0].row()
                    doc = study_list[row]
                    study_data = doc['data']
                    general_data = study_data.get('general', {})
                    
                    preview = f"""
                    <h3>Study Details:</h3>
                    <p><b>ID:</b> {doc['_id']}</p>
                    <p><b>Title:</b> {general_data.get('title', 'Untitled')}</p>
                    <p><b>Type:</b> {general_data.get('study_type', 'Not specified')}</p>
                    <p><b>Last Modified:</b> {doc.get('timestamp', 'Unknown')}</p>
                    <p><b>Description:</b> {general_data.get('description', 'No description')}</p>
                    
                    <h4>Summary:</h4>
                    <ul>
                    <li>Timepoints: {len(study_data.get('timepoints', []))}</li>
                    <li>Arms: {len(study_data.get('arms', []))}</li>
                    <li>Outcomes: {len(study_data.get('outcomes', []))}</li>
                    </ul>
                    """
                    preview_text.setHtml(preview)

            table.itemSelectionChanged.connect(update_preview)
            
            # Add buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            # Show dialog and handle result
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_items = table.selectedItems()
                if selected_items:
                    selected_row = selected_items[0].row()
                    selected_doc = study_list[selected_row]
                    
                    # Log the load action
                    self.db_setup.log_user_action(
                        client_id=self.client_id,
                        action_type="load_study_design",
                        details={
                            "study_id": selected_doc['_id'],
                            "title": selected_doc['data'].get('general', {}).get('title', 'Untitled')
                        }
                    )
                    
                    # Populate the UI with the selected design
                    self.populate_study_data(selected_doc['data'])
                    
                    QMessageBox.information(
                        self, 
                        "Success", 
                        f"Study design '{selected_doc['data'].get('general', {}).get('title', 'Untitled')}' loaded successfully!"
                    )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading study design: {str(e)}")

    def populate_timepoints_data(self, timepoints_data):
        """Populate the timepoints table with loaded data."""
        self.timepoints_table.setRowCount(0)  # Clear existing rows
        for timepoint in timepoints_data:
            row_position = self.timepoints_table.rowCount()
            self.timepoints_table.insertRow(row_position)

            self.timepoints_table.setItem(row_position, 0, QTableWidgetItem(timepoint.get('name', '')))

            # Type Combobox
            type_combo = QComboBox()
            type_combo.addItems([tp.value for tp in TimePoint])
            type_combo.setCurrentText(timepoint.get('type', ''))  # Set directly
            self.timepoints_table.setCellWidget(row_position, 1, type_combo)

            self.timepoints_table.setItem(row_position, 2, QTableWidgetItem(str(timepoint.get('order', ''))))
            self.timepoints_table.setItem(row_position, 3, QTableWidgetItem(str(timepoint.get('offset_days', ''))))
            self.timepoints_table.setItem(row_position, 4, QTableWidgetItem(timepoint.get('description', '')))
            self.timepoints_table.setItem(row_position, 5, QTableWidgetItem(str(timepoint.get('window_days', ''))))
        self.timepoints_table.resizeColumnsToContents()

    def populate_outcomes_data(self, outcomes_data):
        """Populate the outcomes table with loaded data."""
        self.outcomes_table.setRowCount(0)  # Clear existing rows
        for outcome in outcomes_data:
            row_position = self.outcomes_table.rowCount()
            self.outcomes_table.insertRow(row_position)

            self.outcomes_table.setItem(row_position, 0, QTableWidgetItem(outcome.get('name', '')))
            self.outcomes_table.setItem(row_position, 1, QTableWidgetItem(outcome.get('description', '')))
            self.outcomes_table.setItem(row_position, 2, QTableWidgetItem(', '.join(outcome.get('timepoints', []))))

            # Data Type Combobox
            data_type_combo = QComboBox()
            data_type_combo.addItems([dt.value for dt in CFDataType])
            data_type_combo.setCurrentText(outcome.get('data_type', ''))
            self.outcomes_table.setCellWidget(row_position, 3, data_type_combo)

            # Category Combobox
            category_combo = QComboBox()
            category_combo.addItems([oc.value for oc in OutcomeCategory])
            category_combo.setCurrentText(outcome.get('category', ''))
            self.outcomes_table.setCellWidget(row_position, 4, category_combo)

            # Collection Method Combobox
            collection_method_combo = QComboBox()
            collection_method_combo.addItems([cm.value for cm in DataCollectionMethod])
            collection_method_combo.setCurrentText(outcome.get('collection_method', ''))
            self.outcomes_table.setCellWidget(row_position, 5, collection_method_combo)

            self.outcomes_table.setItem(row_position, 6, QTableWidgetItem(', '.join(outcome.get('applicable_arms', []))))
            self.outcomes_table.setItem(row_position, 7, QTableWidgetItem(outcome.get('units', '')))
        self.outcomes_table.resizeColumnsToContents()

    def populate_covariates_data(self, covariates_data):
        """Populate the covariates table with loaded data."""
        self.covariates_table.setRowCount(0)  # Clear existing rows
        for covariate in covariates_data:
            row_position = self.covariates_table.rowCount()
            self.covariates_table.insertRow(row_position)

            self.covariates_table.setItem(row_position, 0, QTableWidgetItem(covariate.get('name', '')))
            self.covariates_table.setItem(row_position, 1, QTableWidgetItem(covariate.get('description', '')))

            # Data Type Combobox
            data_type_combo = QComboBox()
            data_type_combo.addItems([dt.value for dt in CFDataType])
            data_type_combo.setCurrentText(covariate.get('data_type', ''))
            self.covariates_table.setCellWidget(row_position, 2, data_type_combo)
        self.covariates_table.resizeColumnsToContents()

    def populate_randomization_data(self, randomization_data):
        """Populate randomization fields."""
        method = randomization_data.get("method", "")
        index = self.method_combo.findText(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)

        self.block_size_input.setText(str(randomization_data.get("block_size", "")))
        self.stratification_input.setText(", ".join(randomization_data.get("stratification_factors", [])))
        self.randomization_description_input.setPlainText(randomization_data.get("randomization_description", ""))
        self.random_seed_input.setText(str(randomization_data.get("random_seed", "")))
        self.ratio_input.setText(str(randomization_data.get("ratio", "")))

    def populate_blinding_data(self, blinding_data):
        """Populate blinding fields."""
        if not blinding_data:
            return
            
        # Get the blinding type and handle it properly
        blinding_type = blinding_data.get("type", "")
        blinding_type = self._safe_enum_to_str(blinding_type)
            
        # Find and select the blinding type in the combo box
        index = self.blinding_combo.findText(blinding_type)
        if index >= 0:
            self.blinding_combo.setCurrentIndex(index)
            
        # Populate other blinding fields if they exist
        if hasattr(self, 'who_is_blinded_input') and 'who_is_blinded' in blinding_data:
            self.who_is_blinded_input.setText(str(blinding_data.get('who_is_blinded', '')))
            
        if hasattr(self, 'unblinding_procedure_input') and 'unblinding_procedure' in blinding_data:
            self.unblinding_procedure_input.setPlainText(str(blinding_data.get('unblinding_procedure', '')))

    def populate_adverse_events_data(self, adverse_events_data):
        """Populate adverse events table."""
        self.adverse_events_table.setRowCount(0)
        for event in adverse_events_data:
            row_pos = self.adverse_events_table.rowCount()
            self.adverse_events_table.insertRow(row_pos)

            self.adverse_events_table.setItem(row_pos, 0, QTableWidgetItem(event.get("participant_id", "")))
            self.adverse_events_table.setItem(row_pos, 1, QTableWidgetItem(event.get("description", "")))

            severity_combo = QComboBox()
            severity_combo.addItems([sev.value for sev in AdverseEventSeverity])
            severity_combo.setCurrentText(event.get("severity", ""))
            self.adverse_events_table.setCellWidget(row_pos, 2, severity_combo)

            causality_combo = QComboBox()
            causality_combo.addItems([caus.value for caus in AdverseEventCausality])
            causality_combo.setCurrentText(event.get("causality", ""))
            self.adverse_events_table.setCellWidget(row_pos, 3, causality_combo)

            self.adverse_events_table.setItem(row_pos, 4, QTableWidgetItem(event.get("intervention", "")))
            self.adverse_events_table.setItem(row_pos, 5, QTableWidgetItem(event.get("onset_date", "")))
            self.adverse_events_table.setItem(row_pos, 6, QTableWidgetItem(event.get("resolution_date", "")))
            self.adverse_events_table.setItem(row_pos, 7, QTableWidgetItem(event.get("action_taken", "")))
        self.adverse_events_table.resizeColumnsToContents()

    def populate_data_management_data(self, data_management_data):
        """Populate data management fields."""
        self.data_tools_input.setText(", ".join(data_management_data.get("tools", [])))
        self.storage_input.setText(data_management_data.get("storage_location", ""))
        self.security_input.setPlainText(data_management_data.get("security_measures", ""))
        self.sharing_input.setPlainText(data_management_data.get("sharing_policy", ""))
        self.qc_input.setPlainText(data_management_data.get("quality_control", ""))
        self.backup_input.setPlainText(data_management_data.get("backup_procedures", ""))

    def populate_ethics_data(self, ethics_data):
        """Populate ethics fields."""
        self.committee_input.setText(ethics_data.get("committee_name", ""))
        self.approval_id_input.setText(ethics_data.get("approval_id", ""))
        self.approval_date_input.setText(ethics_data.get("approval_date", ""))
        self.expiration_date_input.setText(ethics_data.get("expiration_date", ""))
        self.ethics_notes_input.setPlainText(ethics_data.get("notes", ""))

    def populate_registration_data(self, registration_data):
        """Populate registration fields."""
        self.registry_input.setText(registration_data.get("registry_name", ""))
        self.registration_id_input.setText(registration_data.get("registration_id", ""))
        self.registration_date_input.setText(registration_data.get("registration_date", ""))
        self.url_input.setText(registration_data.get("url", ""))

    def populate_data_collection_data(self, data_collection_data):
        """Populate data collection fields."""
        self.tools_input.setText(", ".join(data_collection_data.get("tools", [])))
        self.procedures_input.setPlainText(data_collection_data.get("procedures", ""))
        self.data_qc_input.setPlainText(data_collection_data.get("quality_control", ""))

    def populate_safety_monitoring_data(self, safety_monitoring_data):
        """Populate safety monitoring fields."""
        self.dsmb_input.setText(", ".join(safety_monitoring_data.get("dsmb_members", [])))
        self.meeting_input.setText(safety_monitoring_data.get("meeting_schedule", ""))
        self.reporting_input.setPlainText(safety_monitoring_data.get("reporting_procedures", ""))


    def reset_all_data(self):
        """Reset all data in the study design section"""
        try:
            # Clear tables
            self.timepoints_table.setRowCount(0)
            self.arms_table.setRowCount(0)
            self.outcomes_table.setRowCount(0)
            self.covariates_table.setRowCount(0)
            self.adverse_events_table.setRowCount(0)
            
            # Reset randomization data
            self.method_combo.setCurrentIndex(0)
            self.block_size_input.clear()
            self.stratification_input.clear()
            self.random_seed_input.clear()
            self.ratio_input.clear()
            
            # Reset blinding
            self.blinding_combo.setCurrentIndex(0)
            
            # Reset data management
            self.data_tools_input.clear()
            self.storage_input.clear()
            self.security_input.clear()
            self.sharing_input.clear()
            self.qc_input.clear()
            self.backup_input.clear()
            
            # Reset ethics
            self.committee_input.clear()
            self.approval_id_input.clear()
            self.approval_date_input.clear()
            self.expiration_date_input.clear()
            self.ethics_notes_input.clear()
            
            # Reset registration
            self.registration_id_input.clear()
            self.registration_date_input.clear()
            
            # Reset safety monitoring
            self.dsmb_input.clear()
            self.meeting_input.clear()
            self.reporting_input.clear()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Some fields could not be reset: {str(e)}")
            
        # Force a repaint
        self.repaint()

    def setup_safety_monitoring_tab(self):
        layout = QVBoxLayout(self.safety_monitoring_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Add a title and description
        title_label = QLabel("Safety Monitoring")
        title_label.setProperty("class", "section-title")  # Use property for main app theming
        description_label = QLabel("Define the procedures and personnel responsible for monitoring participant safety throughout the study.")
        description_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)

        # Create a tabbed interface for different aspects of safety monitoring
        tab_widget = QTabWidget()
        
        # DSMB/Safety Committee Tab
        committee_tab = QWidget()
        committee_layout = QVBoxLayout(committee_tab)
        
        # Create fields for committee info
        committee_fields = []
        
        # Committee Type
        self.committee_type_combo = QComboBox()
        self.committee_type_combo.addItems(["DSMB", "Safety Committee", "Medical Monitor", "Other"])
        committee_fields.append(("Committee Type", self.committee_type_combo))
        
        # Size
        self.committee_size_input = QLineEdit()
        self.committee_size_input.setPlaceholderText("e.g., 3-5 members")
        committee_fields.append(("Committee Size", self.committee_size_input))
        
        # Use compact layout for committee inputs
        committee_form_layout = self.create_compact_form_layout(committee_fields, columns=2)
        committee_layout.addLayout(committee_form_layout)
        
        # DSMB Members
        members_label = QLabel("Committee Members")
        committee_layout.addWidget(members_label)
        
        self.dsmb_input = QLineEdit()
        self.dsmb_input.setPlaceholderText("e.g., Dr. Smith (Chair), Dr. Johnson, Dr. Williams")
        committee_layout.addWidget(self.dsmb_input)
        
        # Meeting Schedule
        meeting_label = QLabel("Meeting Schedule")
        committee_layout.addWidget(meeting_label)
        
        self.meeting_input = QLineEdit()
        self.meeting_input.setPlaceholderText("e.g., Quarterly, or after each 10 participants")
        committee_layout.addWidget(self.meeting_input)
        
        tab_widget.addTab(committee_tab, "Safety Committee")
        
        # Monitoring Procedures Tab
        procedures_tab = QWidget()
        procedures_layout = QVBoxLayout(procedures_tab)
        
        # Stopping Rules
        stopping_label = QLabel("Stopping Rules")
        procedures_layout.addWidget(stopping_label)
        
        self.stopping_rules_input = QTextEdit()
        self.stopping_rules_input.setPlaceholderText("Define criteria that would lead to study termination")
        self.stopping_rules_input.setMaximumHeight(100)
        procedures_layout.addWidget(self.stopping_rules_input)
        
        # Reporting Procedures
        reporting_label = QLabel("Reporting Procedures")
        procedures_layout.addWidget(reporting_label)
        
        self.reporting_input = QTextEdit()
        self.reporting_input.setPlaceholderText("Describe how safety events will be reported, reviewed and acted upon")
        self.reporting_input.setMinimumHeight(100)
        procedures_layout.addWidget(self.reporting_input)
        
        tab_widget.addTab(procedures_tab, "Monitoring Procedures")
        
        layout.addWidget(tab_widget)
        layout.addStretch()
        
    def _safe_enum_to_str(self, value):
        """Safely convert an enum value to a string."""
        if value is None:
            return ""
        if hasattr(value, 'value'):
            return value.value
        return str(value)
