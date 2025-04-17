import pandas as pd
from PyQt6.QtWidgets import (QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QCheckBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, 
                               QMessageBox, QDialog, QDialogButtonBox, QGroupBox, QRadioButton,
                               QDateEdit, QSizePolicy, QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QDate, QEventLoop
from PyQt6.QtGui import QColor, QFont
import asyncio
from PyQt6.QtWidgets import QApplication

from llms.client import call_llm_async_json
from helpers.load_icon import load_bootstrap_icon

class ParticipantManagementSection(QWidget):
    """
    A widget that handles participant management aspects of study design:
    - Eligibility criteria
    - Recruitment
    - Retention
    """
    
    # Define signals that will be used to communicate with parent widget
    data_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None, client_id="default_user", studies_manager=None):
        super().__init__(parent)
        self.client_id = client_id
        self.parent = parent
        self.studies_manager = studies_manager
        
        # Initialize storage for datasets and variables
        self.datasets = {}
        self.inclusion_criteria = []
        self.exclusion_criteria = []
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Setup tabs
        self.setup_eligibility_tab()
        self.setup_recruitment_tab()
        self.setup_retention_tab()
        
        # Load datasets if studies_manager is provided
        if self.studies_manager:
            self.load_datasets_from_active_study()
        
        # Connect tab changed signal to refresh datasets
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    # =========== UTILITY METHODS ===========
    
    def populate_table_from_dataframe(self, table: QTableWidget, df: pd.DataFrame):
        """Populates a QTableWidget from a pandas DataFrame."""
        
        def format_value(val):
            if pd.isna(val):
                return ""
            elif isinstance(val, bool):
                return "Yes" if val else "No"
            elif isinstance(val, (pd.Timestamp, QDate)):
                return val.strftime('%Y-%m-%d')
            elif isinstance(val, (list, tuple)):
                return ', '.join(str(x) for x in val)
            else:
                return str(val)
                
        # Clear the table first
        table.setRowCount(0)
        
        if df is None or df.empty:
            return
            
        # Get column names and set them as headers
        columns = df.columns.tolist()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        
        # Add data
        for row_idx, row in df.iterrows():
            table.insertRow(table.rowCount())
            for col_idx, col_name in enumerate(columns):
                value = row[col_name]
                table.setItem(table.rowCount() - 1, col_idx, 
                              QTableWidgetItem(format_value(value)))
        
        # Make sure table contents fit
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
    
    def table_to_dataframe(self, table: QTableWidget) -> pd.DataFrame:
        """Converts a QTableWidget to a pandas DataFrame."""
        rows = table.rowCount()
        cols = table.columnCount()
        
        if rows == 0 or cols == 0:
            return pd.DataFrame()
            
        # Get header labels
        headers = [table.horizontalHeaderItem(col).text() for col in range(cols)]
        
        # Extract data
        data = []
        for row in range(rows):
            row_data = []
            for col in range(cols):
                item = table.item(row, col)
                if item is None:
                    row_data.append("")
                else:
                    row_data.append(item.text())
            data.append(row_data)
            
        return pd.DataFrame(data, columns=headers)
    
    def configure_table_columns(self, table: QTableWidget, column_proportions=None):
        """Configures the columns of a QTableWidget based on the given proportions."""
        if column_proportions is None:
            # Default to equal proportions
            column_proportions = [1] * table.columnCount()
        
        # Calculate total available width
        available_width = table.width()
        # Account for vertical scrollbar if needed
        if table.verticalScrollBar().isVisible():
            available_width -= table.verticalScrollBar().width()
        
        # Calculate total proportion
        total_proportion = sum(column_proportions)
        
        # Set column widths based on proportions
        for col, proportion in enumerate(column_proportions):
            width = int(available_width * (proportion / total_proportion))
            table.setColumnWidth(col, width)
    
    def adjust_table_columns(self, table, column_proportions):
        """Adjusts table columns using the configure_table_columns method when the table is resized."""
        def resize_event(event):
            self.configure_table_columns(table, column_proportions)
            # Call the original resizeEvent
            QTableWidget.resizeEvent(table, event)
        
        # Replace the table's resizeEvent with our custom one
        table.resizeEvent = resize_event
    
    def get_text_safely(self, widget):
        """Gets text from a widget safely, handling different widget types."""
        if widget is None:
            return ""
        
        if isinstance(widget, QComboBox):
            return widget.currentText()
        elif isinstance(widget, (QLineEdit, QTextEdit)):
            return widget.text() if hasattr(widget, 'text') else widget.toPlainText()
        elif isinstance(widget, QSpinBox):
            return str(widget.value())
        elif isinstance(widget, QCheckBox):
            return "Yes" if widget.isChecked() else "No"
        elif isinstance(widget, QDateEdit):
            return widget.date().toString(Qt.DateFormat.ISODate)
        else:
            return str(widget)
    
    def create_form_layout(self, fields):
        """Creates a form layout from a list of field definitions."""
        form_layout = QFormLayout()
        
        for field_def in fields:
            label = field_def.get("label", "")
            widget = field_def.get("widget")
            
            if widget:
                form_layout.addRow(label, widget)
        
        return form_layout
    
    def create_compact_form_layout(self, fields, columns=2):
        """Creates a compact form layout with multiple columns."""
        layout = QHBoxLayout()
        
        # Calculate how many fields per column
        fields_per_column = (len(fields) + columns - 1) // columns
        
        # Create columns
        for col in range(columns):
            start_idx = col * fields_per_column
            end_idx = min(start_idx + fields_per_column, len(fields))
            
            if start_idx >= len(fields):
                break
                
            column_fields = fields[start_idx:end_idx]
            column_layout = self.create_form_layout(column_fields)
            layout.addLayout(column_layout)
        
        return layout
        
    def set_studies_manager(self, studies_manager):
        """Sets the studies manager for the participant management section."""
        self.studies_manager = studies_manager

    def open_criterion_dialog(self, table, criteria_type):
        """Opens a dialog for adding eligibility criteria."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Add {criteria_type.capitalize()} Criterion")
        
        layout = QVBoxLayout(dialog)
        
        # Form layout for inputs
        form_layout = QFormLayout()
        
        # Description
        description_edit = QTextEdit()
        description_edit.setMaximumHeight(100)
        form_layout.addRow("Description:", description_edit)
        
        # Dataset selection
        dataset_combo = QComboBox()
        dataset_combo.addItem("-- Select Dataset --")
        for dataset_name in self.datasets.keys():
            dataset_combo.addItem(dataset_name)
        form_layout.addRow("Dataset:", dataset_combo)
        
        # Variable selection (populated when dataset is selected)
        variable_combo = QComboBox()
        variable_combo.addItem("-- Select Variable --")
        form_layout.addRow("Variable:", variable_combo)
        
        # Operator
        operator_combo = QComboBox()
        for op in ["=", "!=", ">", "<", ">=", "<=", "contains", "not contains"]:
            operator_combo.addItem(op)
        form_layout.addRow("Operator:", operator_combo)
        
        # Value
        value_edit = QLineEdit()
        form_layout.addRow("Value:", value_edit)
        
        # Notes
        notes_edit = QTextEdit()
        notes_edit.setMaximumHeight(100)
        form_layout.addRow("Notes:", notes_edit)
        
        layout.addLayout(form_layout)
        
        # Connect dataset selection to variable population
        def update_variables():
            variable_combo.clear()
            variable_combo.addItem("-- Select Variable --")
            
            dataset_name = dataset_combo.currentText()
            if dataset_name in self.datasets and dataset_name != "-- Select Dataset --":
                variables = self.datasets[dataset_name].columns.tolist()
                for var in variables:
                    variable_combo.addItem(var)
        
        dataset_combo.currentTextChanged.connect(update_variables)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get values
            description = description_edit.toPlainText()
            dataset = dataset_combo.currentText()
            variable = variable_combo.currentText()
            operator = operator_combo.currentText()
            value = value_edit.text()
            notes = notes_edit.toPlainText()
            
            # Validate
            if dataset == "-- Select Dataset --":
                QMessageBox.warning(self, "Validation Error", "Please select a dataset.")
                return
                
            if variable == "-- Select Variable --":
                QMessageBox.warning(self, "Validation Error", "Please select a variable.")
                return
                
            if not value:
                QMessageBox.warning(self, "Validation Error", "Please enter a value.")
                return
                
            # Add to table
            row = table.rowCount()
            table.insertRow(row)
            
            table.setItem(row, 0, QTableWidgetItem(description))
            table.setItem(row, 1, QTableWidgetItem(dataset))
            table.setItem(row, 2, QTableWidgetItem(variable))
            table.setItem(row, 3, QTableWidgetItem(operator))
            table.setItem(row, 4, QTableWidgetItem(value))
            table.setItem(row, 5, QTableWidgetItem(notes))
            
            # Resize to fit content
            table.resizeRowsToContents()
    
    def remove_criteria_filter(self, table):
        """Removes selected criteria from a table."""
        # Get the selected row(s)
        selected_rows = sorted(set(index.row() for index in table.selectedIndexes()))
        
        # Remove rows in reverse order to avoid index changes
        for row in reversed(selected_rows):
            table.removeRow(row)
    
    def update_dataset_combos(self, section="all"):
        """Updates dataset dropdowns based on available datasets."""
        dataset_names = list(self.datasets.keys())
        print(f"Updating combo boxes with {len(dataset_names)} datasets: {dataset_names}")
        
        # Update different sections based on parameter
        if section in ["all", "eligibility"]:
            # Update eligibility dataset combos
            for combo in [self.inc_dataset_combo, self.exc_dataset_combo, self.validation_dataset_combo]:
                current_text = combo.currentText()
                combo.clear()
                combo.addItem("-- Select Dataset --")
                
                for dataset_name in dataset_names:
                    combo.addItem(dataset_name)
                
                # Try to restore previous selection
                index = combo.findText(current_text)
                if index >= 0:
                    combo.setCurrentIndex(index)
                else:
                    combo.setCurrentIndex(0)  # Default to first item
        
        if section in ["all", "recruitment"]:
            # Update recruitment dataset combos
            current_text = self.recruitment_dataset_combo.currentText()
            self.recruitment_dataset_combo.clear()
            self.recruitment_dataset_combo.addItem("-- Select Dataset --")
            
            for dataset_name in dataset_names:
                self.recruitment_dataset_combo.addItem(dataset_name)
            
            # Try to restore previous selection
            index = self.recruitment_dataset_combo.findText(current_text)
            if index >= 0:
                self.recruitment_dataset_combo.setCurrentIndex(index)
        
        if section in ["all", "retention"]:
            # Update retention dataset combos
            current_text = self.retention_dataset_combo.currentText()
            self.retention_dataset_combo.clear()
            self.retention_dataset_combo.addItem("-- Select Dataset --")
            
            for dataset_name in dataset_names:
                self.retention_dataset_combo.addItem(dataset_name)
            
            # Try to restore previous selection
            index = self.retention_dataset_combo.findText(current_text)
            if index >= 0:
                self.retention_dataset_combo.setCurrentIndex(index)
    
    def on_dataset_changed(self, section):
        """Handles dataset change events for different sections."""
        if section == "eligibility_inc":
            dataset_name = self.inc_dataset_combo.currentText()
            if dataset_name in self.datasets and dataset_name != "-- Select Dataset --":
                variables = self.datasets[dataset_name].columns.tolist()
                self.inc_variable_combo.clear()
                self.inc_variable_combo.addItem("-- Select Variable --")
                for var in variables:
                    self.inc_variable_combo.addItem(var)
            else:
                self.inc_variable_combo.clear()
                self.inc_variable_combo.addItem("-- Select Variable --")
        
        elif section == "eligibility_exc":
            dataset_name = self.exc_dataset_combo.currentText()
            if dataset_name in self.datasets and dataset_name != "-- Select Dataset --":
                variables = self.datasets[dataset_name].columns.tolist()
                self.exc_variable_combo.clear()
                self.exc_variable_combo.addItem("-- Select Variable --")
                for var in variables:
                    self.exc_variable_combo.addItem(var)
            else:
                self.exc_variable_combo.clear()
                self.exc_variable_combo.addItem("-- Select Variable --")
        
        elif section == "recruitment":
            dataset_name = self.recruitment_dataset_combo.currentText()
            if dataset_name in self.datasets and dataset_name != "-- Select Dataset --":
                variables = self.datasets[dataset_name].columns.tolist()
                
                # Update enrollment variable combo
                self.enrollment_var_combo.clear()
                self.enrollment_var_combo.addItem("-- Select Variable --")
                for var in variables:
                    self.enrollment_var_combo.addItem(var)
            else:
                self.enrollment_var_combo.clear()
                self.enrollment_var_combo.addItem("-- Select Variable --")
        
        elif section == "retention":
            dataset_name = self.retention_dataset_combo.currentText()
            if dataset_name in self.datasets and dataset_name != "-- Select Dataset --":
                variables = self.datasets[dataset_name].columns.tolist()
                
                # Update lost to follow-up variable combo
                self.ltf_var_combo.clear()
                self.ltf_var_combo.addItem("-- Select Variable --")
                for var in variables:
                    self.ltf_var_combo.addItem(var)
            else:
                self.ltf_var_combo.clear()
                self.ltf_var_combo.addItem("-- Select Variable --")
    
    def update_variable_dropdowns(self, table):
        """Updates variable dropdowns in a table based on the selected dataset."""
        for row in range(table.rowCount()):
            # Get dataset name from the table
            dataset_item = table.item(row, 1)
            if dataset_item is None:
                continue
                
            dataset_name = dataset_item.text()
            
            # Check if the dataset exists
            if dataset_name not in self.datasets:
                continue
                
            # Get variables for this dataset
            variables = self.datasets[dataset_name].columns.tolist()
            
            # Create a combobox for variable selection
            variable_combo = QComboBox()
            for var in variables:
                variable_combo.addItem(var)
                
            # Get current variable
            variable_item = table.item(row, 2)
            if variable_item is not None:
                current_var = variable_item.text()
                index = variable_combo.findText(current_var)
                if index >= 0:
                    variable_combo.setCurrentIndex(index)
                    
            # Set the combobox as the cell widget
            table.setCellWidget(row, 2, variable_combo)
    
    # =========== ELIGIBILITY TAB ===========
    
    def setup_eligibility_tab(self):
        """Sets up the eligibility criteria tab."""
        eligibility_tab = QWidget()
        layout = QVBoxLayout(eligibility_tab)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Eligibility Criteria")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        description_label = QLabel("Define the inclusion and exclusion criteria for your study.")
        layout.addWidget(description_label)
        layout.addSpacing(10)
        
        # Split inclusion and exclusion into horizontal layout
        criteria_layout = QHBoxLayout()
        
        # ======= INCLUSION SECTION =======
        inclusion_section = QVBoxLayout()
        
        # Dataset selection for inclusion criteria
        dataset_layout_inc = QHBoxLayout()
        dataset_layout_inc.addWidget(QLabel("Dataset for Inclusion Criteria:"))
        self.inc_dataset_combo = QComboBox()
        self.inc_dataset_combo.addItem("-- Select Dataset --")
        self.inc_dataset_combo.currentTextChanged.connect(lambda: self.on_dataset_changed("eligibility_inc"))
        dataset_layout_inc.addWidget(self.inc_dataset_combo)
        dataset_layout_inc.addStretch()
        inclusion_section.addLayout(dataset_layout_inc)
        
        # Inclusion criteria section
        inclusion_group = QGroupBox("Inclusion Criteria")
        inclusion_layout = QVBoxLayout(inclusion_group)
        
        # Table for inclusion criteria
        self.inclusion_table = QTableWidget()
        self.inclusion_table.setColumnCount(6)
        self.inclusion_table.setHorizontalHeaderLabels(
            ["Description", "Dataset", "Variable", "Operator", "Value", "Notes"]
        )
        self.inclusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.inclusion_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        inclusion_layout.addWidget(self.inclusion_table)
        
        # Quick add form in a horizontal layout
        inc_quick_add_layout = QHBoxLayout()
        self.inc_variable_combo = QComboBox()
        self.inc_variable_combo.addItem("-- Select Variable --")
        inc_quick_add_layout.addWidget(QLabel("Variable:"))
        inc_quick_add_layout.addWidget(self.inc_variable_combo)
        
        self.inc_operator_combo = QComboBox()
        for op in ["=", "!=", ">", "<", ">=", "<=", "contains", "not contains"]:
            self.inc_operator_combo.addItem(op)
        inc_quick_add_layout.addWidget(QLabel("Operator:"))
        inc_quick_add_layout.addWidget(self.inc_operator_combo)
        
        self.inc_value_edit = QLineEdit()
        inc_quick_add_layout.addWidget(QLabel("Value:"))
        inc_quick_add_layout.addWidget(self.inc_value_edit)
        
        self.inc_quick_add_btn = QPushButton("Quick Add")
        self.inc_quick_add_btn.setIcon(load_bootstrap_icon("plus-circle"))
        self.inc_quick_add_btn.clicked.connect(lambda: self.add_criteria_filter(
            self.inclusion_table, "inclusion", None, self.inc_variable_combo,
            self.inc_operator_combo, self.inc_value_edit, None
        ))
        inc_quick_add_layout.addWidget(self.inc_quick_add_btn)
        inclusion_layout.addLayout(inc_quick_add_layout)
        
        # Buttons for inclusion criteria
        inc_buttons_layout = QHBoxLayout()
        self.inc_add_btn = QPushButton("Add Criterion")
        self.inc_add_btn.setIcon(load_bootstrap_icon("plus-lg"))
        self.inc_add_btn.clicked.connect(lambda: self.open_criterion_dialog(self.inclusion_table, "inclusion"))
        inc_buttons_layout.addWidget(self.inc_add_btn)
        
        self.inc_remove_btn = QPushButton("Remove Selected")
        self.inc_remove_btn.setIcon(load_bootstrap_icon("trash"))
        self.inc_remove_btn.clicked.connect(lambda: self.remove_criteria_filter(self.inclusion_table))
        inc_buttons_layout.addWidget(self.inc_remove_btn)
        inc_buttons_layout.addStretch()
        inclusion_layout.addLayout(inc_buttons_layout)
        
        # Add natural language input section for inclusion criteria
        nl_inclusion_group = QGroupBox("Natural Language Criteria Input")
        nl_inclusion_layout = QVBoxLayout(nl_inclusion_group)
        
        nl_inclusion_label = QLabel("Describe your inclusion criteria in plain language:")
        nl_inclusion_layout.addWidget(nl_inclusion_label)
        
        self.nl_inclusion_text = QTextEdit()
        self.nl_inclusion_text.setPlaceholderText("Example: Include patients who are over 18 years old, with BMI less than 30, and no history of heart disease")
        self.nl_inclusion_text.setMaximumHeight(100)
        nl_inclusion_layout.addWidget(self.nl_inclusion_text)
        
        nl_inclusion_btn = QPushButton("Process Natural Language Input")
        nl_inclusion_btn.setIcon(load_bootstrap_icon("magic"))
        nl_inclusion_btn.clicked.connect(lambda: self.call_llm_process("inclusion"))
        nl_inclusion_layout.addWidget(nl_inclusion_btn)
        
        inclusion_section.addWidget(nl_inclusion_group)
        
        inclusion_section.addWidget(inclusion_group)
        criteria_layout.addLayout(inclusion_section)
        
        # ======= EXCLUSION SECTION =======
        exclusion_section = QVBoxLayout()
        
        # Dataset selection for exclusion criteria
        dataset_layout_exc = QHBoxLayout()
        dataset_layout_exc.addWidget(QLabel("Dataset for Exclusion Criteria:"))
        self.exc_dataset_combo = QComboBox()
        self.exc_dataset_combo.addItem("-- Select Dataset --")
        self.exc_dataset_combo.currentTextChanged.connect(lambda: self.on_dataset_changed("eligibility_exc"))
        dataset_layout_exc.addWidget(self.exc_dataset_combo)
        dataset_layout_exc.addStretch()
        exclusion_section.addLayout(dataset_layout_exc)
        
        # Exclusion criteria section
        exclusion_group = QGroupBox("Exclusion Criteria")
        exclusion_layout = QVBoxLayout(exclusion_group)
        
        # Table for exclusion criteria
        self.exclusion_table = QTableWidget()
        self.exclusion_table.setColumnCount(6)
        self.exclusion_table.setHorizontalHeaderLabels(
            ["Description", "Dataset", "Variable", "Operator", "Value", "Notes"]
        )
        self.exclusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.exclusion_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        exclusion_layout.addWidget(self.exclusion_table)
        
        # Quick add form
        exc_quick_add_layout = QHBoxLayout()
        self.exc_variable_combo = QComboBox()
        self.exc_variable_combo.addItem("-- Select Variable --")
        exc_quick_add_layout.addWidget(QLabel("Variable:"))
        exc_quick_add_layout.addWidget(self.exc_variable_combo)
        
        self.exc_operator_combo = QComboBox()
        for op in ["=", "!=", ">", "<", ">=", "<=", "contains", "not contains"]:
            self.exc_operator_combo.addItem(op)
        exc_quick_add_layout.addWidget(QLabel("Operator:"))
        exc_quick_add_layout.addWidget(self.exc_operator_combo)
        
        self.exc_value_edit = QLineEdit()
        exc_quick_add_layout.addWidget(QLabel("Value:"))
        exc_quick_add_layout.addWidget(self.exc_value_edit)
        
        self.exc_quick_add_btn = QPushButton("Quick Add")
        self.exc_quick_add_btn.setIcon(load_bootstrap_icon("plus-circle"))
        self.exc_quick_add_btn.clicked.connect(lambda: self.add_criteria_filter(
            self.exclusion_table, "exclusion", None, self.exc_variable_combo,
            self.exc_operator_combo, self.exc_value_edit, None
        ))
        exc_quick_add_layout.addWidget(self.exc_quick_add_btn)
        exclusion_layout.addLayout(exc_quick_add_layout)
        
        # Buttons for exclusion criteria
        exc_buttons_layout = QHBoxLayout()
        self.exc_add_btn = QPushButton("Add Criterion")
        self.exc_add_btn.setIcon(load_bootstrap_icon("plus-lg"))
        self.exc_add_btn.clicked.connect(lambda: self.open_criterion_dialog(self.exclusion_table, "exclusion"))
        exc_buttons_layout.addWidget(self.exc_add_btn)
        
        self.exc_remove_btn = QPushButton("Remove Selected")
        self.exc_remove_btn.setIcon(load_bootstrap_icon("trash"))
        self.exc_remove_btn.clicked.connect(lambda: self.remove_criteria_filter(self.exclusion_table))
        exc_buttons_layout.addWidget(self.exc_remove_btn)
        exc_buttons_layout.addStretch()
        exclusion_layout.addLayout(exc_buttons_layout)
        
        # Add natural language input section for exclusion criteria
        nl_exclusion_group = QGroupBox("Natural Language Criteria Input")
        nl_exclusion_layout = QVBoxLayout(nl_exclusion_group)
        
        nl_exclusion_label = QLabel("Describe your exclusion criteria in plain language:")
        nl_exclusion_layout.addWidget(nl_exclusion_label)
        
        self.nl_exclusion_text = QTextEdit()
        self.nl_exclusion_text.setPlaceholderText("Example: Exclude participants who smoke, are pregnant, or have taken antibiotics in the last 30 days")
        self.nl_exclusion_text.setMaximumHeight(100)
        nl_exclusion_layout.addWidget(self.nl_exclusion_text)
        
        nl_exclusion_btn = QPushButton("Process Natural Language Input")
        nl_exclusion_btn.setIcon(load_bootstrap_icon("magic"))
        nl_exclusion_btn.clicked.connect(lambda: self.call_llm_process("exclusion"))
        nl_exclusion_layout.addWidget(nl_exclusion_btn)
        
        exclusion_section.addWidget(nl_exclusion_group)
        
        exclusion_section.addWidget(exclusion_group)
        criteria_layout.addLayout(exclusion_section)
        
        layout.addLayout(criteria_layout)
        
        # Add eligibility validation section
        eligibility_validation_group = QGroupBox("Validate Eligibility Criteria")
        eligibility_validation_layout = QVBoxLayout(eligibility_validation_group)
        
        # Choose dataset for validation
        validation_dataset_layout = QHBoxLayout()
        validation_dataset_layout.addWidget(QLabel("Dataset for Validation:"))
        self.validation_dataset_combo = QComboBox()
        self.validation_dataset_combo.addItem("-- Select Dataset --")
        for dataset_name in self.datasets.keys():
            self.validation_dataset_combo.addItem(dataset_name)
        validation_dataset_layout.addWidget(self.validation_dataset_combo)
        eligibility_validation_layout.addLayout(validation_dataset_layout)
        
        # Validate button
        self.validate_eligibility_btn = QPushButton("Validate Eligibility Criteria")
        self.validate_eligibility_btn.setIcon(load_bootstrap_icon("check-circle"))
        self.validate_eligibility_btn.clicked.connect(self.validate_eligibility_criteria)
        eligibility_validation_layout.addWidget(self.validate_eligibility_btn)
        
        # Results area
        self.eligibility_results = QTextEdit()
        self.eligibility_results.setReadOnly(True)
        self.eligibility_results.setMaximumHeight(150)
        eligibility_validation_layout.addWidget(self.eligibility_results)
        
        layout.addWidget(eligibility_validation_group)
        
        # Refresh button aligned to the right
        refresh_btn = QPushButton("Refresh Datasets")
        refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_btn.clicked.connect(self.refresh_datasets)
        refresh_btn_layout = QHBoxLayout()
        refresh_btn_layout.addStretch()
        refresh_btn_layout.addWidget(refresh_btn)
        layout.addLayout(refresh_btn_layout)
        
        self.tab_widget.addTab(eligibility_tab, "Eligibility")
    
    def add_criteria_filter(self, table, criteria_type, desc_edit=None, var_combo=None, 
                           operator_combo=None, value_edit=None, notes_edit=None):
        """Adds a criteria filter to the specified table."""
        # Get values
        description = desc_edit.toPlainText() if desc_edit else ""
        variable = var_combo.currentText() if var_combo else ""
        operator = operator_combo.currentText() if operator_combo else "="
        value = value_edit.text() if value_edit else ""
        notes = notes_edit.toPlainText() if notes_edit else ""
        
        # Get dataset
        if criteria_type == "inclusion":
            dataset = self.inc_dataset_combo.currentText()
        elif criteria_type == "exclusion":
            dataset = self.exc_dataset_combo.currentText()
        else:
            dataset = ""
        
        # Validation
        if dataset == "-- Select Dataset --" or not dataset:
            QMessageBox.warning(self, "Validation Error", "Please select a dataset.")
            return
            
        if variable == "-- Select Variable --" or not variable:
            QMessageBox.warning(self, "Validation Error", "Please select a variable.")
            return
            
        if not value:
            QMessageBox.warning(self, "Validation Error", "Please enter a value.")
            return
            
        # Add to table
        row = table.rowCount()
        table.insertRow(row)
        
        table.setItem(row, 0, QTableWidgetItem(description))
        table.setItem(row, 1, QTableWidgetItem(dataset))
        table.setItem(row, 2, QTableWidgetItem(variable))
        table.setItem(row, 3, QTableWidgetItem(operator))
        table.setItem(row, 4, QTableWidgetItem(value))
        table.setItem(row, 5, QTableWidgetItem(notes))
        
        # Clear the form
        if desc_edit:
            desc_edit.clear()
        if value_edit:
            value_edit.clear()
        if notes_edit:
            notes_edit.clear()
        if var_combo:
            var_combo.setCurrentIndex(0)
            
        # Resize rows to contents
        table.resizeRowsToContents()
    
    async def process_natural_language_criteria(self, criteria_type):
        """Process natural language input to create structured criteria."""
        # Get appropriate text and dataset based on criteria type
        if criteria_type == 'inclusion':
            text = self.nl_inclusion_text.toPlainText().strip()
            dataset_name = self.inc_dataset_combo.currentText()
            target_table = self.inclusion_table
        elif criteria_type == 'exclusion':
            text = self.nl_exclusion_text.toPlainText().strip()
            dataset_name = self.exc_dataset_combo.currentText()
            target_table = self.exclusion_table
        elif criteria_type == 'enrollment':
            text = self.nl_enrollment_text.toPlainText().strip()
            dataset_name = self.recruitment_dataset_combo.currentText()
            target_table = self.enrollment_table
        elif criteria_type == 'ltf':
            text = self.nl_ltf_text.toPlainText().strip()
            dataset_name = self.retention_dataset_combo.currentText()
            target_table = self.ltf_table
        else:
            QMessageBox.warning(self, "Error", "Invalid criteria type specified.")
            return
        
        # Validate input
        if not text:
            QMessageBox.warning(self, "Validation Error", "Please enter a description of your criteria.")
            return
        
        if dataset_name == "-- Select Dataset --" or dataset_name not in self.datasets:
            QMessageBox.warning(self, "Validation Error", "Please select a valid dataset first.")
            return
        
        # Create a progress dialog instead of a message box
        progress_dialog = QProgressDialog("Processing natural language input...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Processing")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)  # Make it modal to prevent other interactions
        progress_dialog.setCancelButton(None)  # Remove cancel button
        progress_dialog.setMinimumDuration(0)  # Show immediately
        progress_dialog.setAutoClose(True)     # Automatically close when done
        progress_dialog.show()
        
        # Process application events to ensure dialog appears
        QApplication.processEvents()
        
        result = None
        try:
            # Get the dataset
            df = self.datasets[dataset_name]
            
            # Get columns and sample values to provide context to the LLM
            columns = df.columns.tolist()
            
            # Generate column information with sample values
            column_info = []
            for col in columns:
                col_data = {
                    "name": col,
                    "dtype": str(df[col].dtype)
                }
                
                # For categorical/object columns, provide distinct values (limited to reasonable number)
                if df[col].dtype.kind in 'OSU':  # Object, String, Unicode
                    unique_values = df[col].dropna().unique().tolist()
                    # Limit to 15 values to avoid making the prompt too large
                    sample_values = unique_values[:15] if len(unique_values) <= 15 else unique_values[:10] + ["..."]
                    col_data["sample_values"] = sample_values
                    if len(unique_values) > 15:
                        col_data["value_count"] = len(unique_values)
                
                # For numeric columns, provide range information
                elif df[col].dtype.kind in 'ifc':  # Integer, Float, Complex
                    try:
                        col_data["min"] = float(df[col].min())
                        col_data["max"] = float(df[col].max())
                        col_data["mean"] = float(df[col].mean())
                        # Add some sample values for context
                        col_data["sample_values"] = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
                    except:
                        # Handle any errors in calculating stats
                        pass
                
                # For date columns, provide range information
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        col_data["min"] = df[col].min().strftime('%Y-%m-%d')
                        col_data["max"] = df[col].max().strftime('%Y-%m-%d')
                        # Add some sample dates
                        col_data["sample_values"] = [d.strftime('%Y-%m-%d') for d in 
                                                  df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()]
                    except:
                        # Handle any errors
                        pass
                    
                # For boolean columns, show True/False counts
                elif df[col].dtype.kind == 'b':
                    try:
                        value_counts = df[col].value_counts().to_dict()
                        col_data["true_count"] = int(value_counts.get(True, 0))
                        col_data["false_count"] = int(value_counts.get(False, 0))
                        col_data["sample_values"] = [True, False]
                    except:
                        # Handle any errors
                        pass
                
                column_info.append(col_data)
            
            # Format the prompt as a string
            prompt_str = f"""
            I need to convert a natural language description of {criteria_type} criteria into structured format.
            
            Dataset: {dataset_name}
            Available columns:
            """
            
            # Add column information in a readable format
            for col_data in column_info:
                prompt_str += f"\n- {col_data['name']} ({col_data['dtype']})"
                
                if 'sample_values' in col_data:
                    sample_values = str(col_data['sample_values']).replace("'", '"')
                    prompt_str += f"\n  Sample values: {sample_values}"
                
                if 'min' in col_data and 'max' in col_data:
                    prompt_str += f"\n  Range: {col_data['min']} to {col_data['max']}"
            
            # Add the criteria description from user
            prompt_str += f"\n\nCriteria description: {text}\n"
            
            # Specify expected output format
            prompt_str += """
            Please respond with a JSON object having the following structure:
            {
                "criteria": [
                    {
                        "description": "Brief description of the criterion",
                        "variable": "Column name from dataset - must be one of the provided columns",
                        "operator": "One of: =, !=, >, <, >=, <=, contains, not contains",
                        "value": "Value to compare against, should be appropriate for the column data type",
                        "notes": "Any additional notes"
                    }
                ],
                "explanation": "Explanation of how the criteria were interpreted"
            }
            
            Ensure you only use column names that exist in the dataset and provide values that match the column data types.
            """
            
            # Call the LLM with a string prompt
            result = await call_llm_async_json(prompt_str)
            
        except Exception as e:
            print(f"LLM processing error details: {type(e).__name__}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process natural language input: {str(e)}")
            return
        finally:
            # Explicitly close and destroy the progress dialog
            progress_dialog.close()
            progress_dialog.deleteLater()
            QApplication.processEvents()
        
        # Only proceed if we have a valid result
        if result:
            # Show explanation dialog as a separate function for better handling
            self.show_criteria_explanation(result, criteria_type, target_table, dataset_name)

    def show_criteria_explanation(self, result, criteria_type, target_table, dataset_name):
        """Show explanation dialog and handle user confirmation."""
        explanation_dialog = QMessageBox(self)
        explanation_dialog.setWindowTitle("Criteria Interpretation")
        explanation_dialog.setText(result["explanation"])
        explanation_dialog.setDetailedText("The following criteria were extracted:\n" + 
                                "\n".join([f"- {c['description']}: {c['variable']} {c['operator']} {c['value']}" 
                                        for c in result["criteria"]]))
        explanation_dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        
        # If user confirms, add the criteria to the table
        if explanation_dialog.exec() == QMessageBox.StandardButton.Ok:
            for criterion in result["criteria"]:
                row = target_table.rowCount()
                target_table.insertRow(row)
                
                # Map criterion fields to appropriate table columns based on criteria_type
                if criteria_type in ['inclusion', 'exclusion']:
                    # Eligibility tables have 6 columns: ["Description", "Dataset", "Variable", "Operator", "Value", "Notes"]
                    target_table.setItem(row, 0, QTableWidgetItem(criterion["description"]))
                    target_table.setItem(row, 1, QTableWidgetItem(dataset_name))
                    target_table.setItem(row, 2, QTableWidgetItem(criterion["variable"]))
                    target_table.setItem(row, 3, QTableWidgetItem(criterion["operator"]))
                    target_table.setItem(row, 4, QTableWidgetItem(criterion["value"]))
                    target_table.setItem(row, 5, QTableWidgetItem(criterion.get("notes", "")))
                elif criteria_type in ['enrollment', 'ltf']:
                    # These tables have 4 columns: ["Variable", "Condition", "Value", "Notes"]
                    target_table.setItem(row, 0, QTableWidgetItem(criterion["variable"]))
                    target_table.setItem(row, 1, QTableWidgetItem(criterion["operator"]))  # "operator" → "Condition"
                    target_table.setItem(row, 2, QTableWidgetItem(criterion["value"]))
                    
                    # Use description + notes for the Notes column
                    notes_text = criterion.get("description", "")
                    if criterion.get("notes"):
                        if notes_text:
                            notes_text += " - "
                        notes_text += criterion.get("notes")
                    target_table.setItem(row, 3, QTableWidgetItem(notes_text))
                
                # Clear the text input
                if criteria_type == 'inclusion':
                    self.nl_inclusion_text.clear()
                elif criteria_type == 'exclusion':
                    self.nl_exclusion_text.clear()
                elif criteria_type == 'enrollment':
                    self.nl_enrollment_text.clear()
                elif criteria_type == 'ltf':
                    self.nl_ltf_text.clear()
            
            # Resize rows to fit content
            target_table.resizeRowsToContents()
            
            QMessageBox.information(self, "Success", f"Added {len(result['criteria'])} criteria to the table.")
    
    def call_llm_process(self, criteria_type):
        """
        Wrapper to handle async call to process_natural_language_criteria
        """
        loop = QEventLoop()
        asyncio.create_task(self._run_process(criteria_type, loop))
        loop.exec()
        
    async def _run_process(self, criteria_type, loop):
        """Helper method to run the async process and close the event loop when done"""
        try:
            await self.process_natural_language_criteria(criteria_type)
        finally:
            loop.quit()
    
    def validate_eligibility_criteria(self):
        """Validates eligibility criteria against the selected dataset."""
        dataset_name = self.validation_dataset_combo.currentText()
        
        if dataset_name == "-- Select Dataset --" or dataset_name not in self.datasets:
            QMessageBox.warning(self, "Validation Error", "Please select a valid dataset for validation.")
            return
        
        if self.inclusion_table.rowCount() == 0 and self.exclusion_table.rowCount() == 0:
            QMessageBox.warning(self, "Validation Error", "Please define at least one inclusion or exclusion criterion.")
            return
        
        df = self.datasets[dataset_name]
        total_rows = len(df)
        
        # Create case-insensitive column lookup dictionary
        column_lookup = {col.lower(): col for col in df.columns}
        
        # Initialize results tracking
        results_text = f"Validation results using dataset: {dataset_name}\n"
        results_text += f"Total records in dataset: {total_rows}\n\n"
        
        # Apply inclusion criteria
        include_df = df.copy()
        inclusion_results = []
        
        if self.inclusion_table.rowCount() > 0:
            results_text += "INCLUSION CRITERIA:\n"
            
            for row in range(self.inclusion_table.rowCount()):
                description = self.inclusion_table.item(row, 0).text() if self.inclusion_table.item(row, 0) else ""
                dataset = self.inclusion_table.item(row, 1).text() if self.inclusion_table.item(row, 1) else ""
                variable = self.inclusion_table.item(row, 2).text() if self.inclusion_table.item(row, 2) else ""
                operator = self.inclusion_table.item(row, 3).text() if self.inclusion_table.item(row, 3) else ""
                value = self.inclusion_table.item(row, 4).text() if self.inclusion_table.item(row, 4) else ""
                
                if not all([dataset, variable, operator, value]):
                    continue
                    
                # If the dataset doesn't match current validation dataset, skip
                if dataset != dataset_name:
                    inclusion_results.append(f"⚠️ Criterion '{description}' uses different dataset ({dataset}) - skipped")
                    continue
            
                # Find the actual column name using case-insensitive matching
                actual_variable = None
                if variable in df.columns:
                    actual_variable = variable
                elif variable.lower() in column_lookup:
                    actual_variable = column_lookup[variable.lower()]
                    inclusion_results.append(f"ℹ️ Used '{actual_variable}' instead of '{variable}' (case-insensitive match)")
                
                if not actual_variable:
                    inclusion_results.append(f"❌ Variable '{variable}' not found in dataset")
                    continue
                
                # Clean and prepare the value for comparison
                clean_value = value.strip()
                
                # Convert value to appropriate type based on column data type
                try:
                    if include_df[actual_variable].dtype.kind in 'ifc':  # numeric types
                        try:
                            clean_value = float(clean_value)
                        except ValueError:
                            inclusion_results.append(f"❌ Cannot convert '{value}' to number for column '{actual_variable}'")
                            continue
                    elif include_df[actual_variable].dtype.kind in 'bB':  # boolean types
                        if clean_value.lower() in ('true', 'yes', '1', 't', 'y'):
                            clean_value = True
                        elif clean_value.lower() in ('false', 'no', '0', 'f', 'n'):
                            clean_value = False
                except Exception as e:
                    inclusion_results.append(f"❌ Error handling value '{value}': {str(e)}")
                    continue
                
                # Apply filter
                original_count = len(include_df)
                try:
                    if operator == "=":
                        mask = include_df[actual_variable] == clean_value
                    elif operator == "!=":
                        mask = include_df[actual_variable] != clean_value
                    elif operator == ">":
                        mask = include_df[actual_variable] > clean_value
                    elif operator == "<":
                        mask = include_df[actual_variable] < clean_value
                    elif operator == ">=":
                        mask = include_df[actual_variable] >= clean_value
                    elif operator == "<=":
                        mask = include_df[actual_variable] <= clean_value
                    elif operator == "contains":
                        # Convert both sides to string for contains operation
                        mask = include_df[actual_variable].astype(str).str.contains(str(clean_value), case=False, na=False)
                    elif operator == "not contains":
                        mask = ~include_df[actual_variable].astype(str).str.contains(str(clean_value), case=False, na=False)
                    else:
                        inclusion_results.append(f"❌ Unsupported operator: {operator}")
                        continue
                    
                    # Handle NaN values - exclude them from results
                    if mask.isna().any():
                        mask = mask.fillna(False)
                    
                    include_df = include_df[mask]
                    matching = len(include_df)
                    percent = (matching / original_count) * 100 if original_count > 0 else 0
                    
                    criterion_msg = f"{description if description else actual_variable + ' ' + operator + ' ' + str(clean_value)}"
                    inclusion_results.append(f"✓ {criterion_msg}: {matching}/{original_count} ({percent:.1f}%)")
                    
                except Exception as e:
                    inclusion_results.append(f"❌ Error applying {actual_variable} {operator} {clean_value}: {str(e)}")
            
            # Add inclusion results
            results_text += "\n".join(inclusion_results)
            results_text += f"\n\nAfter all inclusion criteria: {len(include_df)}/{total_rows} ({(len(include_df)/total_rows)*100:.1f}% eligible)\n\n"
        
        # Apply exclusion criteria to the results of inclusion
        exclude_df = include_df.copy()
        exclusion_results = []
        
        if self.exclusion_table.rowCount() > 0:
            results_text += "EXCLUSION CRITERIA:\n"
            
            for row in range(self.exclusion_table.rowCount()):
                description = self.exclusion_table.item(row, 0).text() if self.exclusion_table.item(row, 0) else ""
                dataset = self.exclusion_table.item(row, 1).text() if self.exclusion_table.item(row, 1) else ""
                variable = self.exclusion_table.item(row, 2).text() if self.exclusion_table.item(row, 2) else ""
                operator = self.exclusion_table.item(row, 3).text() if self.exclusion_table.item(row, 3) else ""
                value = self.exclusion_table.item(row, 4).text() if self.exclusion_table.item(row, 4) else ""
                
                if not all([dataset, variable, operator, value]):
                    continue
                    
                # If the dataset doesn't match current validation dataset, skip
                if dataset != dataset_name:
                    exclusion_results.append(f"⚠️ Criterion '{description}' uses different dataset ({dataset}) - skipped")
                    continue
            
                # Find the actual column name using case-insensitive matching
                actual_variable = None
                if variable in df.columns:
                    actual_variable = variable
                elif variable.lower() in column_lookup:
                    actual_variable = column_lookup[variable.lower()]
                    exclusion_results.append(f"ℹ️ Used '{actual_variable}' instead of '{variable}' (case-insensitive match)")
                
                if not actual_variable:
                    exclusion_results.append(f"❌ Variable '{variable}' not found in dataset")
                    continue
                
                # Clean and prepare the value for comparison
                clean_value = value.strip()
                
                # Convert value to appropriate type based on column data type
                try:
                    if exclude_df[actual_variable].dtype.kind in 'ifc':  # numeric types
                        try:
                            clean_value = float(clean_value)
                        except ValueError:
                            exclusion_results.append(f"❌ Cannot convert '{value}' to number for column '{actual_variable}'")
                            continue
                    elif exclude_df[actual_variable].dtype.kind in 'bB':  # boolean types
                        if clean_value.lower() in ('true', 'yes', '1', 't', 'y'):
                            clean_value = True
                        elif clean_value.lower() in ('false', 'no', '0', 'f', 'n'):
                            clean_value = False
                except Exception as e:
                    exclusion_results.append(f"❌ Error handling value '{value}': {str(e)}")
                    continue
                
                # Apply filter (for exclusion criteria, we want to KEEP records that DON'T match)
                original_count = len(exclude_df)
                try:
                    # First create a mask of records that MATCH the criterion (to be excluded)
                    if operator == "=":
                        exclude_mask = exclude_df[actual_variable] == clean_value
                    elif operator == "!=":
                        exclude_mask = exclude_df[actual_variable] != clean_value
                    elif operator == ">":
                        exclude_mask = exclude_df[actual_variable] > clean_value
                    elif operator == "<":
                        exclude_mask = exclude_df[actual_variable] < clean_value
                    elif operator == ">=":
                        exclude_mask = exclude_df[actual_variable] >= clean_value
                    elif operator == "<=":
                        exclude_mask = exclude_df[actual_variable] <= clean_value
                    elif operator == "contains":
                        exclude_mask = exclude_df[actual_variable].astype(str).str.contains(str(clean_value), case=False, na=False)
                    elif operator == "not contains":
                        exclude_mask = ~exclude_df[actual_variable].astype(str).str.contains(str(clean_value), case=False, na=False)
                    else:
                        exclusion_results.append(f"❌ Unsupported operator: {operator}")
                        continue
                    
                    # Handle NaN values - consider them as not matching
                    if exclude_mask.isna().any():
                        exclude_mask = exclude_mask.fillna(False)
                    
                    # Count records that would be excluded
                    to_exclude_count = exclude_mask.sum()
                    
                    # Keep records that DON'T match the exclusion criterion
                    exclude_df = exclude_df[~exclude_mask]
                    
                    remaining = len(exclude_df)
                    percent_excluded = (to_exclude_count / original_count) * 100 if original_count > 0 else 0
                    
                    criterion_msg = f"{description if description else actual_variable + ' ' + operator + ' ' + str(clean_value)}"
                    exclusion_results.append(f"✓ {criterion_msg}: excluded {to_exclude_count}/{original_count} ({percent_excluded:.1f}%)")
                    
                except Exception as e:
                    exclusion_results.append(f"❌ Error applying {actual_variable} {operator} {clean_value}: {str(e)}")
            
            # Add exclusion results
            results_text += "\n".join(exclusion_results)
            results_text += f"\n\nAfter all exclusion criteria: {len(exclude_df)}/{total_rows} ({(len(exclude_df)/total_rows)*100:.1f}% eligible)\n\n"
        
        # Final eligible count
        eligible_count = len(exclude_df)
        eligible_percent = (eligible_count / total_rows) * 100 if total_rows > 0 else 0
        
        results_text += f"SUMMARY:\n"
        results_text += f"Total eligible: {eligible_count}/{total_rows} ({eligible_percent:.1f}%)"
        
        # Display results
        self.eligibility_results.setText(results_text)
    
    # =========== RECRUITMENT TAB ===========
    
    def setup_recruitment_tab(self):
        """Sets up the recruitment tab."""
        recruitment_tab = QWidget()
        layout = QVBoxLayout(recruitment_tab)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Recruitment")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Top section with recruitment period and methods side by side
        top_section = QHBoxLayout()
        
        # Recruitment period
        period_group = QGroupBox("Recruitment Period")
        period_layout = QFormLayout(period_group)
        
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate())
        period_layout.addRow("Start Date:", self.start_date_edit)
        
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate().addYears(1))
        period_layout.addRow("End Date:", self.end_date_edit)
        
        self.expected_sample_spin = QSpinBox()
        self.expected_sample_spin.setRange(1, 1000000)
        self.expected_sample_spin.setValue(100)
        period_layout.addRow("Expected Sample Size:", self.expected_sample_spin)
        
        top_section.addWidget(period_group)
        
        # Recruitment methods
        methods_group = QGroupBox("Recruitment Methods")
        methods_layout = QVBoxLayout(methods_group)
        
        self.recruitment_methods = {
            "print_ads": QCheckBox("Print advertisements"),
            "social_media": QCheckBox("Social media"),
            "direct_contact": QCheckBox("Direct contact"),
            "referral": QCheckBox("Referral from clinicians"),
            "database": QCheckBox("Existing participant database"),
            "community": QCheckBox("Community outreach"),
            "other": QCheckBox("Other")
        }
        
        for method, checkbox in self.recruitment_methods.items():
            methods_layout.addWidget(checkbox)
        
        self.other_method_edit = QLineEdit()
        self.other_method_edit.setPlaceholderText("Specify other recruitment method")
        self.other_method_edit.setEnabled(False)
        methods_layout.addWidget(self.other_method_edit)
        
        # Connect the "Other" checkbox to enable/disable the text field
        self.recruitment_methods["other"].toggled.connect(self.other_method_edit.setEnabled)
        
        top_section.addWidget(methods_group)
        layout.addLayout(top_section)
        
        # Enrollment tracking
        enrollment_group = QGroupBox("Enrollment Tracking")
        enrollment_layout = QVBoxLayout(enrollment_group)
        
        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset:"))
        self.recruitment_dataset_combo = QComboBox()
        self.recruitment_dataset_combo.addItem("-- Select Dataset --")
        self.recruitment_dataset_combo.currentTextChanged.connect(lambda: self.on_dataset_changed("recruitment"))
        dataset_layout.addWidget(self.recruitment_dataset_combo)
        dataset_layout.addStretch()
        enrollment_layout.addLayout(dataset_layout)
        
        # Split the enrollment section into table and results horizontally
        enrollment_content = QHBoxLayout()
        
        # Left side - enrollment table and controls
        enrollment_left = QVBoxLayout()
        
        # Enrollment status table
        self.enrollment_table = QTableWidget()
        self.enrollment_table.setColumnCount(4)
        self.enrollment_table.setHorizontalHeaderLabels(
            ["Variable", "Condition", "Value", "Notes"]
        )
        self.enrollment_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.enrollment_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        enrollment_left.addWidget(self.enrollment_table)
        
        # Options in a horizontal layout
        options_layout = QHBoxLayout()
        self.enrollment_var_combo = QComboBox()
        self.enrollment_var_combo.addItem("-- Select Variable --")
        options_layout.addWidget(QLabel("Variable:"))
        options_layout.addWidget(self.enrollment_var_combo)
        
        self.enrollment_op_combo = QComboBox()
        for op in ["=", "!=", ">", "<", ">=", "<=", "contains", "not contains"]:
            self.enrollment_op_combo.addItem(op)
        options_layout.addWidget(QLabel("Condition:"))
        options_layout.addWidget(self.enrollment_op_combo)
        
        self.enrollment_value_edit = QLineEdit()
        options_layout.addWidget(QLabel("Value:"))
        options_layout.addWidget(self.enrollment_value_edit)
        options_layout.addStretch()
        enrollment_left.addLayout(options_layout)
        
        # Add/remove enrollment status
        buttons_layout = QHBoxLayout()
        
        self.add_enrollment_btn = QPushButton("Add")
        self.add_enrollment_btn.setIcon(load_bootstrap_icon("plus-lg"))
        self.add_enrollment_btn.clicked.connect(self.add_enrollment_status)
        buttons_layout.addWidget(self.add_enrollment_btn)
        
        self.remove_enrollment_btn = QPushButton("Remove Selected")
        self.remove_enrollment_btn.setIcon(load_bootstrap_icon("trash"))
        self.remove_enrollment_btn.clicked.connect(lambda: self.remove_criteria_filter(self.enrollment_table))
        buttons_layout.addWidget(self.remove_enrollment_btn)
        buttons_layout.addStretch()
        enrollment_left.addLayout(buttons_layout)
        
        # Add natural language input for enrollment criteria
        nl_enrollment_group = QGroupBox("Natural Language Enrollment Criteria")
        nl_enrollment_layout = QVBoxLayout(nl_enrollment_group)
        
        nl_enrollment_label = QLabel("Describe your enrollment criteria in plain language:")
        nl_enrollment_layout.addWidget(nl_enrollment_label)
        
        self.nl_enrollment_text = QTextEdit()
        self.nl_enrollment_text.setPlaceholderText("Example: Track participants who have completed the baseline survey and signed the consent form")
        self.nl_enrollment_text.setMaximumHeight(100)
        nl_enrollment_layout.addWidget(self.nl_enrollment_text)
        
        nl_enrollment_btn = QPushButton("Process Natural Language Input")
        nl_enrollment_btn.setIcon(load_bootstrap_icon("magic"))
        nl_enrollment_btn.clicked.connect(lambda: self.call_llm_process("enrollment"))
        nl_enrollment_layout.addWidget(nl_enrollment_btn)
        
        enrollment_left.addWidget(nl_enrollment_group)
        
        enrollment_content.addLayout(enrollment_left, 3)  # 3:1 ratio
        
        # Right side - verification and results
        enrollment_right = QVBoxLayout()
        
        # Verify button
        self.verify_enrollment_btn = QPushButton("Verify Enrollment Criteria")
        self.verify_enrollment_btn.setIcon(load_bootstrap_icon("check-circle"))
        self.verify_enrollment_btn.clicked.connect(self.verify_enrollment)
        enrollment_right.addWidget(self.verify_enrollment_btn)
        
        # Results
        self.enrollment_results = QTextEdit()
        self.enrollment_results.setReadOnly(True)
        enrollment_right.addWidget(self.enrollment_results)
        
        # Notes
        notes_group = QGroupBox("Recruitment Notes")
        notes_layout = QVBoxLayout(notes_group)
        self.recruitment_notes = QTextEdit()
        notes_layout.addWidget(self.recruitment_notes)
        enrollment_right.addWidget(notes_group)
        
        enrollment_content.addLayout(enrollment_right, 1)  # 3:1 ratio
        
        enrollment_layout.addLayout(enrollment_content)
        layout.addWidget(enrollment_group)
        
        # Refresh button aligned right
        refresh_btn = QPushButton("Refresh Datasets")
        refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_btn.clicked.connect(self.refresh_datasets)
        refresh_btn_layout = QHBoxLayout()
        refresh_btn_layout.addStretch()
        refresh_btn_layout.addWidget(refresh_btn)
        layout.addLayout(refresh_btn_layout)
        
        self.tab_widget.addTab(recruitment_tab, "Recruitment")
    
    def add_enrollment_status(self):
        """Adds enrollment status criteria to the enrollment table."""
        variable = self.enrollment_var_combo.currentText()
        operator = self.enrollment_op_combo.currentText()
        value = self.enrollment_value_edit.text()
        
        # Validation
        if variable == "-- Select Variable --" or not variable:
            QMessageBox.warning(self, "Validation Error", "Please select a variable.")
            return
            
        if not value:
            QMessageBox.warning(self, "Validation Error", "Please enter a value.")
            return
            
        row = self.enrollment_table.rowCount()
        self.enrollment_table.insertRow(row)
        self.enrollment_table.setItem(row, 0, QTableWidgetItem(variable))
        self.enrollment_table.setItem(row, 1, QTableWidgetItem(operator))
        self.enrollment_table.setItem(row, 2, QTableWidgetItem(value))
        self.enrollment_table.setItem(row, 3, QTableWidgetItem(""))
        
        self.enrollment_var_combo.setCurrentIndex(0)
        self.enrollment_value_edit.clear()
        self.enrollment_table.resizeRowsToContents()
    
    def verify_enrollment(self):
        """Verifies enrollment criteria against the selected dataset."""
        dataset_name = self.recruitment_dataset_combo.currentText()
        
        if dataset_name == "-- Select Dataset --" or dataset_name not in self.datasets:
            QMessageBox.warning(self, "Validation Error", "Please select a valid dataset.")
            return
            
        if self.enrollment_table.rowCount() == 0:
            QMessageBox.warning(self, "Validation Error", "Please add at least one enrollment criterion.")
            return
            
        df = self.datasets[dataset_name]
        total_rows = len(df)
        filtered_df = df.copy()
        criteria_descriptions = []
        
        for row in range(self.enrollment_table.rowCount()):
            variable_item = self.enrollment_table.item(row, 0)
            operator_item = self.enrollment_table.item(row, 1)
            value_item = self.enrollment_table.item(row, 2)
            
            if not all([variable_item, operator_item, value_item]):
                continue
                
            variable = variable_item.text()
            operator = operator_item.text()
            value = value_item.text()
            
            if variable not in filtered_df.columns:
                criteria_descriptions.append(f"Variable '{variable}' not found in dataset")
                continue
                
            try:
                if filtered_df[variable].dtype.kind in 'ifc':
                    value = float(value)
            except ValueError:
                pass
                
            try:
                if operator == "=":
                    mask = filtered_df[variable] == value
                elif operator == "!=":
                    mask = filtered_df[variable] != value
                elif operator == ">":
                    mask = filtered_df[variable] > value
                elif operator == "<":
                    mask = filtered_df[variable] < value
                elif operator == ">=":
                    mask = filtered_df[variable] >= value
                elif operator == "<=":
                    mask = filtered_df[variable] <= value
                elif operator == "contains":
                    mask = filtered_df[variable].astype(str).str.contains(str(value), na=False)
                elif operator == "not contains":
                    mask = ~filtered_df[variable].astype(str).str.contains(str(value), na=False)
                else:
                    criteria_descriptions.append(f"Unsupported operator: {operator}")
                    continue
                    
                filtered_df = filtered_df[mask]
                criteria_descriptions.append(f"{variable} {operator} {value}: {len(filtered_df)} matches")
                
            except Exception as e:
                criteria_descriptions.append(f"Error applying {variable} {operator} {value}: {str(e)}")
        
        matching_count = len(filtered_df)
        percentage = (matching_count / total_rows) * 100 if total_rows > 0 else 0
        
        results = (
            f"Total records: {total_rows}\n"
            f"Matching records: {matching_count} ({percentage:.1f}%)\n\n"
            "Filter results:\n" + "\n".join(criteria_descriptions)
        )
        
        self.enrollment_results.setText(results)
    
    # =========== RETENTION TAB ===========
    
    def setup_retention_tab(self):
        """Sets up the retention tab."""
        retention_tab = QWidget()
        layout = QVBoxLayout(retention_tab)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Retention and Attrition")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        description_label = QLabel("Track and analyze participant retention and attrition.")
        layout.addWidget(description_label)
        layout.addSpacing(10)
        
        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset:"))
        self.retention_dataset_combo = QComboBox()
        self.retention_dataset_combo.addItem("-- Select Dataset --")
        self.retention_dataset_combo.currentTextChanged.connect(lambda: self.on_dataset_changed("retention"))
        dataset_layout.addWidget(self.retention_dataset_combo)
        dataset_layout.addStretch()
        layout.addLayout(dataset_layout)
        
        # Split the retention section into two columns
        retention_content = QHBoxLayout()
        
        # Left side - Lost to follow-up criteria
        ltf_group = QGroupBox("Lost to Follow-up Criteria")
        ltf_layout = QVBoxLayout(ltf_group)
        
        self.ltf_table = QTableWidget()
        self.ltf_table.setColumnCount(4)
        self.ltf_table.setHorizontalHeaderLabels(
            ["Variable", "Condition", "Value", "Notes"]
        )
        self.ltf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.ltf_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        ltf_layout.addWidget(self.ltf_table)
        
        # Form for adding criteria
        ltf_form_layout = QHBoxLayout()
        self.ltf_var_combo = QComboBox()
        self.ltf_var_combo.addItem("-- Select Variable --")
        ltf_form_layout.addWidget(QLabel("Variable:"))
        ltf_form_layout.addWidget(self.ltf_var_combo)
        
        self.ltf_op_combo = QComboBox()
        for op in ["=", "!=", ">", "<", ">=", "<=", "contains", "not contains"]:
            self.ltf_op_combo.addItem(op)
        ltf_form_layout.addWidget(QLabel("Condition:"))
        ltf_form_layout.addWidget(self.ltf_op_combo)
        
        self.ltf_value_edit = QLineEdit()
        ltf_form_layout.addWidget(QLabel("Value:"))
        ltf_form_layout.addWidget(self.ltf_value_edit)
        ltf_layout.addLayout(ltf_form_layout)
        
        # Notes field
        notes_layout = QHBoxLayout()
        self.ltf_notes_edit = QLineEdit()
        notes_layout.addWidget(QLabel("Notes:"))
        notes_layout.addWidget(self.ltf_notes_edit)
        ltf_layout.addLayout(notes_layout)
        
        # Buttons for LTF criteria
        ltf_buttons_layout = QHBoxLayout()
        self.add_ltf_btn = QPushButton("Add Criterion")
        self.add_ltf_btn.setIcon(load_bootstrap_icon("plus-lg"))
        self.add_ltf_btn.clicked.connect(self.add_attrition_criterion)
        ltf_buttons_layout.addWidget(self.add_ltf_btn)
        
        self.remove_ltf_btn = QPushButton("Remove Selected")
        self.remove_ltf_btn.setIcon(load_bootstrap_icon("trash"))
        self.remove_ltf_btn.clicked.connect(lambda: self.remove_criteria_filter(self.ltf_table))
        ltf_buttons_layout.addWidget(self.remove_ltf_btn)
        ltf_buttons_layout.addStretch()
        ltf_layout.addLayout(ltf_buttons_layout)
        
        # Check status button
        self.check_ltf_btn = QPushButton("Check Lost to Follow-up Status")
        self.check_ltf_btn.setIcon(load_bootstrap_icon("search"))
        self.check_ltf_btn.clicked.connect(self.check_ltf_status)
        ltf_layout.addWidget(self.check_ltf_btn)
        
        # Results
        self.ltf_results = QTextEdit()
        self.ltf_results.setReadOnly(True)
        self.ltf_results.setMaximumHeight(100)
        ltf_layout.addWidget(self.ltf_results)
        
        # Add natural language input section for lost to follow-up criteria
        nl_ltf_group = QGroupBox("Natural Language Lost to Follow-up Criteria")
        nl_ltf_layout = QVBoxLayout(nl_ltf_group)
        
        nl_ltf_label = QLabel("Describe your lost to follow-up criteria in plain language:")
        nl_ltf_layout.addWidget(nl_ltf_label)
        
        self.nl_ltf_text = QTextEdit()
        self.nl_ltf_text.setPlaceholderText("Example: Consider participants lost to follow-up if they missed two consecutive visits or haven't responded to contact attempts in 60 days")
        self.nl_ltf_text.setMaximumHeight(100)
        nl_ltf_layout.addWidget(self.nl_ltf_text)
        
        nl_ltf_btn = QPushButton("Process Natural Language Input")
        nl_ltf_btn.setIcon(load_bootstrap_icon("magic"))
        nl_ltf_btn.clicked.connect(lambda: self.call_llm_process("ltf"))
        nl_ltf_layout.addWidget(nl_ltf_btn)
        
        ltf_layout.addWidget(nl_ltf_group)
        
        retention_content.addWidget(ltf_group)
        
        # Right side - Retention strategies and analysis
        retention_right = QVBoxLayout()
        
        # Retention strategies
        strategies_group = QGroupBox("Retention Strategies")
        strategies_layout = QVBoxLayout(strategies_group)
        
        # Split checkboxes into two columns
        strategies_columns = QHBoxLayout()
        left_strategies = QVBoxLayout()
        right_strategies = QVBoxLayout()
        
        self.retention_strategies = {
            "reminders": QCheckBox("Appointment reminders"),
            "incentives": QCheckBox("Participant incentives"),
            "follow_up": QCheckBox("Regular follow-up contact"),
            "community": QCheckBox("Community engagement"),
            "flexible": QCheckBox("Flexible scheduling"),
            "transportation": QCheckBox("Transportation assistance"),
            "other": QCheckBox("Other")
        }
        
        # Put first half in left column, second half in right column
        strategy_keys = list(self.retention_strategies.keys())
        middle_index = len(strategy_keys) // 2
        
        for i, (strategy, checkbox) in enumerate(self.retention_strategies.items()):
            if i < middle_index:
                left_strategies.addWidget(checkbox)
            else:
                right_strategies.addWidget(checkbox)
        
        strategies_columns.addLayout(left_strategies)
        strategies_columns.addLayout(right_strategies)
        strategies_layout.addLayout(strategies_columns)
        
        self.other_strategy_edit = QLineEdit()
        self.other_strategy_edit.setPlaceholderText("Specify other retention strategy")
        self.other_strategy_edit.setEnabled(False)
        strategies_layout.addWidget(self.other_strategy_edit)
        self.retention_strategies["other"].toggled.connect(self.other_strategy_edit.setEnabled)
        
        retention_right.addWidget(strategies_group)
        
        # Attrition analysis
        attrition_group = QGroupBox("Attrition Analysis")
        attrition_layout = QVBoxLayout(attrition_group)
        self.attrition_analysis_text = QTextEdit()
        self.attrition_analysis_text.setPlaceholderText("Describe your plan for analyzing attrition and its potential impact on the study.")
        attrition_layout.addWidget(self.attrition_analysis_text)
        retention_right.addWidget(attrition_group)
        
        retention_content.addLayout(retention_right)
        layout.addLayout(retention_content)
        
        # Refresh button aligned right
        refresh_btn = QPushButton("Refresh Datasets")
        refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_btn.clicked.connect(self.refresh_datasets)
        refresh_btn_layout = QHBoxLayout()
        refresh_btn_layout.addStretch()
        refresh_btn_layout.addWidget(refresh_btn)
        layout.addLayout(refresh_btn_layout)
        
        self.tab_widget.addTab(retention_tab, "Retention")
    
    def add_attrition_criterion(self):
        """Adds attrition criterion to the lost to follow-up table."""
        variable = self.ltf_var_combo.currentText()
        operator = self.ltf_op_combo.currentText()
        value = self.ltf_value_edit.text()
        notes = self.ltf_notes_edit.text()
        
        if variable == "-- Select Variable --" or not variable:
            QMessageBox.warning(self, "Validation Error", "Please select a variable.")
            return
            
        if not value:
            QMessageBox.warning(self, "Validation Error", "Please enter a value.")
            return
            
        row = self.ltf_table.rowCount()
        self.ltf_table.insertRow(row)
        self.ltf_table.setItem(row, 0, QTableWidgetItem(variable))
        self.ltf_table.setItem(row, 1, QTableWidgetItem(operator))
        self.ltf_table.setItem(row, 2, QTableWidgetItem(value))
        self.ltf_table.setItem(row, 3, QTableWidgetItem(notes))
        
        self.ltf_var_combo.setCurrentIndex(0)
        self.ltf_value_edit.clear()
        self.ltf_notes_edit.clear()
        self.ltf_table.resizeRowsToContents()
    
    def check_ltf_status(self):
        """Checks the lost to follow-up status against the selected dataset."""
        dataset_name = self.retention_dataset_combo.currentText()
        
        if dataset_name == "-- Select Dataset --" or dataset_name not in self.datasets:
            QMessageBox.warning(self, "Validation Error", "Please select a valid dataset.")
            return
            
        if self.ltf_table.rowCount() == 0:
            QMessageBox.warning(self, "Validation Error", "Please add at least one lost to follow-up criterion.")
            return
            
        df = self.datasets[dataset_name]
        total_rows = len(df)
        filtered_df = df.copy()
        criteria_descriptions = []
        
        for row in range(self.ltf_table.rowCount()):
            variable_item = self.ltf_table.item(row, 0)
            operator_item = self.ltf_table.item(row, 1)
            value_item = self.ltf_table.item(row, 2)
            
            if not all([variable_item, operator_item, value_item]):
                continue
                
            variable = variable_item.text()
            operator = operator_item.text()
            value = value_item.text()
            
            if variable not in filtered_df.columns:
                criteria_descriptions.append(f"Variable '{variable}' not found in dataset")
                continue
                
            try:
                if filtered_df[variable].dtype.kind in 'ifc':
                    value = float(value)
            except ValueError:
                pass
                
            try:
                if operator == "=":
                    mask = filtered_df[variable] == value
                elif operator == "!=":
                    mask = filtered_df[variable] != value
                elif operator == ">":
                    mask = filtered_df[variable] > value
                elif operator == "<":
                    mask = filtered_df[variable] < value
                elif operator == ">=":
                    mask = filtered_df[variable] >= value
                elif operator == "<=":
                    mask = filtered_df[variable] <= value
                elif operator == "contains":
                    mask = filtered_df[variable].astype(str).str.contains(str(value), na=False)
                elif operator == "not contains":
                    mask = ~filtered_df[variable].astype(str).str.contains(str(value), na=False)
                else:
                    criteria_descriptions.append(f"Unsupported operator: {operator}")
                    continue
                    
                filtered_df = filtered_df[mask]
                criteria_descriptions.append(f"{variable} {operator} {value}: {len(filtered_df)} matches")
                
            except Exception as e:
                criteria_descriptions.append(f"Error applying {variable} {operator} {value}: {str(e)}")
        
        matching_count = len(filtered_df)
        percentage = (matching_count / total_rows) * 100 if total_rows > 0 else 0
        
        results = (
            f"Total participants: {total_rows}\n"
            f"Lost to follow-up: {matching_count} ({percentage:.1f}%)\n\n"
            "Filter results:\n" + "\n".join(criteria_descriptions)
        )
        
        self.ltf_results.setText(results)
    
    # =========== DATA COLLECTION AND SAVING ===========
    
    def collect_eligibility_data(self):
        """Collects eligibility criteria data from UI components."""
        inclusion_df = self.table_to_dataframe(self.inclusion_table)
        exclusion_df = self.table_to_dataframe(self.exclusion_table)
        
        return {
            "inclusion_criteria": inclusion_df.to_dict(orient="records") if not inclusion_df.empty else [],
            "exclusion_criteria": exclusion_df.to_dict(orient="records") if not exclusion_df.empty else []
        }
    
    def collect_recruitment_data(self):
        """Collects recruitment data from UI components."""
        methods = {}
        for method, checkbox in self.recruitment_methods.items():
            methods[method] = checkbox.isChecked()
        
        if methods.get("other", False):
            methods["other_text"] = self.other_method_edit.text()
        
        enrollment_df = self.table_to_dataframe(self.enrollment_table)
        
        return {
            "start_date": self.start_date_edit.date().toString(Qt.DateFormat.ISODate),
            "end_date": self.end_date_edit.date().toString(Qt.DateFormat.ISODate),
            "expected_sample_size": self.expected_sample_spin.value(),
            "methods": methods,
            "dataset": self.recruitment_dataset_combo.currentText(),
            "enrollment_criteria": enrollment_df.to_dict(orient="records") if not enrollment_df.empty else [],
            "notes": self.recruitment_notes.toPlainText()
        }
    
    def collect_retention_data(self):
        """Collects retention and attrition data from UI components."""
        strategies = {}
        for strategy, checkbox in self.retention_strategies.items():
            strategies[strategy] = checkbox.isChecked()
        
        if strategies.get("other", False):
            strategies["other_text"] = self.other_strategy_edit.text()
        
        ltf_df = self.table_to_dataframe(self.ltf_table)
        
        return {
            "dataset": self.retention_dataset_combo.currentText(),
            "ltf_criteria": ltf_df.to_dict(orient="records") if not ltf_df.empty else [],
            "strategies": strategies,
            "attrition_analysis": self.attrition_analysis_text.toPlainText()
        }
    
    def collect_data(self):
        """Collects all data from the participant management section."""
        return {
            "eligibility": self.collect_eligibility_data(),
            "recruitment": self.collect_recruitment_data(),
            "retention": self.collect_retention_data()
        }
    
    def populate_eligibility_data(self, eligibility_data):
        """Populates eligibility criteria from saved data."""
        if not eligibility_data:
            return
            
        inclusion_criteria = eligibility_data.get("inclusion_criteria", [])
        if inclusion_criteria:
            inclusion_df = pd.DataFrame(inclusion_criteria)
            self.populate_table_from_dataframe(self.inclusion_table, inclusion_df)
        
        exclusion_criteria = eligibility_data.get("exclusion_criteria", [])
        if exclusion_criteria:
            exclusion_df = pd.DataFrame(exclusion_criteria)
            self.populate_table_from_dataframe(self.exclusion_table, exclusion_df)
    
    def populate_recruitment_data(self, recruitment_data):
        """Populates recruitment data from saved data."""
        if not recruitment_data:
            return
            
        if "start_date" in recruitment_data:
            try:
                self.start_date_edit.setDate(QDate.fromString(recruitment_data["start_date"], Qt.DateFormat.ISODate))
            except:
                pass
                
        if "end_date" in recruitment_data:
            try:
                self.end_date_edit.setDate(QDate.fromString(recruitment_data["end_date"], Qt.DateFormat.ISODate))
            except:
                pass
                
        if "expected_sample_size" in recruitment_data:
            self.expected_sample_spin.setValue(int(recruitment_data["expected_sample_size"]))
        
        methods = recruitment_data.get("methods", {})
        for method, checkbox in self.recruitment_methods.items():
            checkbox.setChecked(methods.get(method, False))
            
        if methods.get("other", False) and "other_text" in methods:
            self.other_method_edit.setText(methods["other_text"])
        
        dataset = recruitment_data.get("dataset", "")
        index = self.recruitment_dataset_combo.findText(dataset)
        if index >= 0:
            self.recruitment_dataset_combo.setCurrentIndex(index)
        
        enrollment_criteria = recruitment_data.get("enrollment_criteria", [])
        if enrollment_criteria:
            enrollment_df = pd.DataFrame(enrollment_criteria)
            self.populate_table_from_dataframe(self.enrollment_table, enrollment_df)
        
        self.recruitment_notes.setPlainText(recruitment_data.get("notes", ""))
    
    def populate_retention_data(self, retention_data):
        """Populates retention data from saved data."""
        if not retention_data:
            return
            
        dataset = retention_data.get("dataset", "")
        index = self.retention_dataset_combo.findText(dataset)
        if index >= 0:
            self.retention_dataset_combo.setCurrentIndex(index)
        
        ltf_criteria = retention_data.get("ltf_criteria", [])
        if ltf_criteria:
            ltf_df = pd.DataFrame(ltf_criteria)
            self.populate_table_from_dataframe(self.ltf_table, ltf_df)
        
        strategies = retention_data.get("strategies", {})
        for strategy, checkbox in self.retention_strategies.items():
            checkbox.setChecked(strategies.get(strategy, False))
            
        if strategies.get("other", False) and "other_text" in strategies:
            self.other_strategy_edit.setText(strategies["other_text"])
        
        self.attrition_analysis_text.setPlainText(retention_data.get("attrition_analysis", ""))
    
    def populate_data(self, data):
        """Populates all data in the participant management section."""
        if not data:
            return
            
        self.populate_eligibility_data(data.get("eligibility", {}))
        self.populate_recruitment_data(data.get("recruitment", {}))
        self.populate_retention_data(data.get("retention", {}))
    
    def load_datasets_from_active_study(self):
        """Loads datasets from the active study in StudiesManager."""
        if not self.studies_manager:
            print("StudiesManager not available")
            return
        
        print("Loading datasets from studies_manager...")
        active_study = self.studies_manager.get_active_study()
        print(f"Active study: {active_study.name if active_study else 'None'}")
        
        datasets = {}
        dataset_tuples = self.studies_manager.get_datasets_from_active_study()
        
        print(f"Got {len(dataset_tuples)} datasets from active study")
        print(f"Dataset types: {[type(d) for d in dataset_tuples]}")
        
        for i, dataset in enumerate(dataset_tuples):
            print(f"Processing dataset {i}...")
            print(f"  Type: {type(dataset)}")
            print(f"  Content: {dataset}")
            
            if isinstance(dataset, dict):
                name = dataset.get('name')
                dataframe = dataset.get('data')
                print(f"  Dict format - name: {name}, dataframe type: {type(dataframe)}")
            else:
                try:
                    name, dataframe = dataset
                    print(f"  Tuple format - name: {name}, dataframe type: {type(dataframe)}")
                except Exception as e:
                    print(f"  Error unpacking: {e}")
                    continue
            
            if name and isinstance(dataframe, pd.DataFrame):
                print(f"  Adding dataset: {name} with {len(dataframe)} rows")
                datasets[name] = dataframe
            else:
                print(f"  Skipping dataset - name valid: {bool(name)}, dataframe is DataFrame: {isinstance(dataframe, pd.DataFrame)}")
        
        print(f"Final datasets dictionary has {len(datasets)} datasets: {list(datasets.keys())}")
        self.datasets = datasets
        self.update_dataset_combos()
    
    def set_datasets(self, datasets):
        """Sets the available datasets and updates UI components."""
        if not datasets:
            print("No datasets provided to set_datasets")
            return
        
        if isinstance(datasets, list):
            datasets_dict = {}
            for name, dataframe in datasets:
                datasets_dict[name] = dataframe
            self.datasets = datasets_dict
        else:
            self.datasets = datasets
        
        print(f"Setting {len(self.datasets)} datasets: {', '.join(self.datasets.keys())}")
        self.update_dataset_combos()
    
    def reset_data(self):
        """Resets all data in the participant management section."""
        self.inclusion_table.setRowCount(0)
        self.exclusion_table.setRowCount(0)
        
        self.start_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setDate(QDate.currentDate().addYears(1))
        self.expected_sample_spin.setValue(100)
        
        for checkbox in self.recruitment_methods.values():
            checkbox.setChecked(False)
        self.other_method_edit.clear()
        self.other_method_edit.setEnabled(False)
        
        self.enrollment_table.setRowCount(0)
        self.recruitment_notes.clear()
        self.enrollment_results.clear()
        
        self.ltf_table.setRowCount(0)
        
        for checkbox in self.retention_strategies.values():
            checkbox.setChecked(False)
        self.other_strategy_edit.clear()
        self.other_strategy_edit.setEnabled(False)
        
        self.attrition_analysis_text.clear()
        self.ltf_results.clear()
        
        self.inc_dataset_combo.setCurrentIndex(0)
        self.exc_dataset_combo.setCurrentIndex(0)
        self.recruitment_dataset_combo.setCurrentIndex(0)
        self.retention_dataset_combo.setCurrentIndex(0)
        
        self.inc_variable_combo.clear()
        self.inc_variable_combo.addItem("-- Select Variable --")
        self.exc_variable_combo.clear()
        self.exc_variable_combo.addItem("-- Select Variable --")
        self.enrollment_var_combo.clear()
        self.enrollment_var_combo.addItem("-- Select Variable --")
        self.ltf_var_combo.clear()
        self.ltf_var_combo.addItem("-- Select Variable --")
    
    def refresh_datasets(self, datasets=None):
        """Refreshes the datasets and updates UI components."""
        if self.studies_manager and datasets is None:
            self.load_datasets_from_active_study()
        elif datasets is not None:
            self.set_datasets(datasets)
        
        self.update_dataset_combos()
        self.data_updated.emit(self.collect_data())
    
    def emit_data_updated(self):
        """Emits the data_updated signal with current data."""
        self.data_updated.emit(self.collect_data())

    def on_tab_changed(self, index):
        """Refreshes datasets when tab is changed"""
        if self.studies_manager:
            print(f"Tab changed to {index}, refreshing datasets")
            self.refresh_datasets()
